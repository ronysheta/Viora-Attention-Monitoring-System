import time
import json
import threading
import requests
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────────────
# Change these values to adjust session behaviour

TTS_ENDPOINT = "https://8000-01kp5wfqkr31z35ytgdv2waqcr.cloudspaces.litng.ai/tts"

POMODORO = {
    "focus_minutes":            25,
    "short_break_minutes":       5,
    "long_break_minutes":       15,
    "blocks_before_long_break":  4,
}

FREE = {
    "focus_minutes":  60,
    "break_minutes":  10,
}

ATTENTION = {
    "inactivity_threshold_seconds": 90,
    "response_window_seconds":      15,
}

MESSAGES = {
    "session_start":  "Your focus session has started. Good luck!",
    "block_start":    "Focus block started. I will let you know when time is up.",
    "block_end":      "Great work! Block complete. Say 1 for a break, 2 to keep going, or 3 to end the session.",
    "break_start":    "Break time. Relax. I will let you know when break time is up.",
    "break_end":      "Break is over. Press any key when you are ready for the next block.",
    "session_end":    "Session complete. Well done!",
    "summary_intro":  "Here is your session summary.",
    "distraction":    "It seems you drifted off. Let us refocus.",
    "checkin":        "Session paused. Come back when you are ready.",
    "resume":         "Welcome back! Resuming your session now.",
}

SUMMARIES_DIR = "summaries"


# TTS 
def speak(text: str):
    """Send text to the TTS endpoint."""
    print(f"[TTS] {text}")
    try:
        requests.post(TTS_ENDPOINT, json={"text": text}, timeout=10)
    except Exception as e:
        print(f"[TTS] Error: {e}")


# Session states 
IDLE      = "idle"
FOCUSING  = "focusing"
BLOCK_END = "block_end"
ON_BREAK  = "on_break"
ENDED     = "ended"


# Focus Session
class FocusSession:
    """
    Runs a focus session in either Pomodoro or Free mode.

    Integrates with AttentionMonitor for distraction detection.
    Saves a JSON summary at the end and reads it aloud.

    Usage:
        session = FocusSession(mode="pomodoro")
        session.start()

        # Call on every user interaction (key press, pause, rewind, etc.)
        session.register_interaction()

        # Call with user's choice at the end of each block
        session.user_choice("break")   # or "continue" or "end"
    """

    def __init__(
        self,
        mode: str            = "pomodoro",
        focus_minutes: int   = None,
        break_minutes: int   = None,
        monitor_mode: str    = "camera",   # "camera" or "no_camera"
        camera_source        = 0,              # int or DroidCam URL string
        frame_source         = None,           # optional shared frame callable
        remaining_time      = None,
    ):
        assert mode in ("pomodoro", "free"), "mode must be 'pomodoro' or 'free'"
        self.mode = mode

        # Override config defaults if user specified custom durations
        if focus_minutes:
            if mode == "pomodoro":
                POMODORO["focus_minutes"] = focus_minutes
            else:
                FREE["focus_minutes"] = focus_minutes
        if break_minutes:
            if mode == "pomodoro":
                POMODORO["short_break_minutes"] = break_minutes
            else:
                FREE["break_minutes"] = break_minutes
        self.state = IDLE

        # Stats tracked during session
        self._blocks_completed   = 0
        self._breaks_taken       = 0
        self._distraction_count  = 0
        self._total_focus_secs   = 0
        self._total_break_secs   = 0
        self._distraction_times  = []
        self._started_at         = None
        self._block_start        = None
        self._break_start        = None

        # Internal timer
        self._timer_stop  = threading.Event()
        self._timer_thread = None

        # Attention monitor — swap by changing monitor_mode
        if monitor_mode == "camera":
            from camera_attention_monitor import CameraAttentionMonitor
            self._monitor = CameraAttentionMonitor(
                on_distraction=self._on_distraction,
                on_resume=self._on_resume,
                speak_fn=speak,
                inactivity_threshold=ATTENTION["inactivity_threshold_seconds"],
                checkin_message=MESSAGES["checkin"],
                alert_message=MESSAGES["resume"],
                camera_source=camera_source,
                frame_source=frame_source,
            )
        else:
            from attention_monitor import AttentionMonitor
            self._monitor = AttentionMonitor(
                on_distraction=self._on_distraction,
                speak_fn=speak,
                inactivity_threshold=ATTENTION["inactivity_threshold_seconds"],
                response_window=ATTENTION["response_window_seconds"],
                checkin_message=MESSAGES["checkin"],
                alert_message=MESSAGES["distraction"],
            )

    # Public API 

    def start(self):
        """Start the session."""
        self._started_at = datetime.now().isoformat()
        speak(MESSAGES["session_start"])
        self._start_focus_block()

    def register_interaction(self):
        """Call this on any user input to reset the inactivity timer."""
        self._monitor.register_interaction()

    def user_choice(self, choice: str):
        """
        Call with the user's response at block end.
        Accepts: "1"/"break", "2"/"continue", "3"/"end"
        """
        if self.state != BLOCK_END:
            return

        choice = choice.strip().lower()

        if choice in ("1", "break"):
            self._start_break()
        elif choice in ("2", "continue"):
            self._start_focus_block()
        elif choice in ("3", "end"):
            self._end_session()
        else:
            speak("Sorry, I didn't catch that. Press 1 for break, 2 to continue, or 3 to end.")

    def stop(self):
        """Force stop from outside (e.g. app close)."""
        self._end_session(forced=True)

    # State transitions
    def _start_focus_block(self):
        self._set_state(FOCUSING)
        self._block_start = time.time()
        speak(MESSAGES["block_start"])

        if self._blocks_completed == 0:
            self._monitor.start()
        else:
            self._monitor.register_interaction()

        self._start_timer(self._focus_duration(), self._on_block_end)

    def _on_block_end(self):
        if self.state != FOCUSING:
            return

        self._total_focus_secs += int(time.time() - self._block_start)
        self._blocks_completed += 1
        self._monitor.register_interaction()
        self._set_state(BLOCK_END)
        speak(MESSAGES["block_end"])

    def _start_break(self):
        self._set_state(ON_BREAK)
        self._break_start = time.time()
        self._breaks_taken += 1
        speak(MESSAGES["break_start"])
        self._start_timer(self._break_duration(), self._on_break_end)

    def _on_break_end(self):
        if self.state != ON_BREAK:
            return

        self._total_break_secs += int(time.time() - self._break_start)
        speak(MESSAGES["break_end"])
        self._start_focus_block()

    def _end_session(self, forced: bool = False):
        self._set_state(ENDED)
        self._stop_timer()
        self._monitor.stop()

        if not forced:
            speak(MESSAGES["session_end"])

        self._save_and_read_summary()
        self._set_state(IDLE)

    # Pause
    def _pause_session(self):
        print("[FocusSession] Pausing session due to distraction.")

        # Calculate remaining time BEFORE stopping timer
        elapsed = int(time.time() - self._block_start)
        total   = self._focus_duration()
        self.remaining_time = max(total - elapsed, 10)

        # Stop timer
        self._stop_timer()

        # Change state
        self._set_state("paused")

        speak("Session paused. Look back to continue.")

    # Distraction 

    def _on_distraction(self):
        if self.state != FOCUSING:
            return
        self._distraction_count += 1
        self._distraction_times.append(datetime.now().isoformat())
        print(f"[FocusSession] Distraction #{self._distraction_count} recorded.")
        self._pause_session()

    def _on_resume(self):
        if self.state != "paused":
            return
        if self.remaining_time is None:
            print("[FocusSession] No remaining time found, restarting block.")
            self._start_focus_block()
            return
        print("[FocusSession] User returned. Resuming block timer.")

        self._block_start = time.time()
        self._start_timer(self.remaining_time, self._on_block_end)

        self._set_state(FOCUSING)

    # ── Summary ───────────────────────────────────────────────────────────────

    def _save_and_read_summary(self):
        import os
        os.makedirs(SUMMARIES_DIR, exist_ok=True)

        summary = {
            "mode":                  self.mode,
            "started_at":            self._started_at,
            "ended_at":              datetime.now().isoformat(),
            "blocks_completed":      self._blocks_completed,
            "breaks_taken":          self._breaks_taken,
            "distraction_count":     self._distraction_count,
            "distraction_timestamps": self._distraction_times,
            "total_focus_minutes":   round(self._total_focus_secs / 60, 1),
            "total_break_minutes":   round(self._total_break_secs / 60, 1),
        }

        filename = f"{SUMMARIES_DIR}/session_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[FocusSession] Summary saved → {filename}")

        # Read summary aloud
        speak(MESSAGES["summary_intro"])
        speak(self._summary_as_speech(summary))

    def _summary_as_speech(self, s: dict) -> str:
        lines = [
            f"You completed {s['blocks_completed']} focus block{'s' if s['blocks_completed'] != 1 else ''}.",
            f"Total focus time: {s['total_focus_minutes']} minutes.",
            f"You took {s['breaks_taken']} break{'s' if s['breaks_taken'] != 1 else ''}.",
        ]
        if s["distraction_count"] == 0:
            lines.append("No distractions detected. Excellent focus!")
        elif s["distraction_count"] == 1:
            lines.append("One distraction was detected.")
        else:
            lines.append(f"{s['distraction_count']} distractions were detected.")
        return " ".join(lines)

    # ── Timer helpers ─────────────────────────────────────────────────────────

    def _start_timer(self, seconds: int, callback):
        self._stop_timer()
        self._timer_stop.clear()

        def run():
            self._timer_stop.wait(timeout=seconds)
            if not self._timer_stop.is_set():
                callback()

        self._timer_thread = threading.Thread(target=run, daemon=True)
        self._timer_thread.start()

    def _stop_timer(self):
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_stop.set()
            self._timer_thread.join(timeout=2)

    def _focus_duration(self) -> int:
        if self.mode == "pomodoro":
            return POMODORO["focus_minutes"] * 60
        return FREE["focus_minutes"] * 60

    def _break_duration(self) -> int:
        if self.mode == "pomodoro":
            if self._blocks_completed % POMODORO["blocks_before_long_break"] == 0:
                return POMODORO["long_break_minutes"] * 60
            return POMODORO["short_break_minutes"] * 60
        return FREE["break_minutes"] * 60

    def _set_state(self, new_state: str):
        print(f"[FocusSession] {self.state} → {new_state}")
        self.state = new_state


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Shorten everything for testing
    POMODORO["focus_minutes"]       = 0.1   # ~6 seconds
    POMODORO["short_break_minutes"] = 0.05
    ATTENTION["inactivity_threshold_seconds"] = 8
    ATTENTION["response_window_seconds"]      = 5

    session = FocusSession(mode="pomodoro")
    session.start()

    try:
        while True:
            key = input("Press Enter to interact, type b/c/e for block-end choice: ").strip()
            if key in ("b", "c", "e"):
                session.user_choice({"b": "break", "c": "continue", "e": "end"}[key])
            else:
                session.register_interaction()
    except KeyboardInterrupt:
        session.stop()
