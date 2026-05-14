"""
focus_session.py
────────────────
Central coordinator for Viora focus sessions.

Manages all session states (focusing, break, paused, ended),
owns the block timer, creates the right attention monitor based
on the user's accessibility profile, and wires everything together.

Changes from v1:
  - Accepts accessibility_profile and stt_fn — wired through to monitors
  - Auto-selects monitor type based on profile (blind → no camera)
  - Camera failure → transparent fallback to no-camera mode
  - All messages from accessibility_profile.PROMPTS in Egyptian Arabic
  - block_end choice now voice-driven via parse_block_end_choice()
  - summary_as_speech in Egyptian Arabic
  - logging throughout instead of print
  - user_profile loaded and saved via accessibility_profile helpers
  - remaining_time initialised properly to avoid AttributeError

Usage:
    from focus_session import FocusSession

    session = FocusSession(
        mode                  = "pomodoro",
        focus_minutes         = 25,
        break_minutes         = 5,
        accessibility_profile = "standard",
        speak_fn              = tts.speak,
        camera_source         = 0,
    )
    session.start()
    session.register_interaction()   # call on any key press
    session.user_choice("استراحة")   # or "كمل" or "خلاص"
    session.stop()
"""

import os
import time
import json
import logging
import threading
from datetime import datetime
from typing import Callable, Optional

from accessibility_profile import (
    PROMPTS,
    PROFILES,
    get_profile,
    load_user_profile,
    save_user_profile,
    parse_block_end_choice,
)

logger = logging.getLogger(__name__)


# ── TTS endpoint ───────────────────────────────────────────────────────────────
# Loaded from environment — never hardcoded.

import os as _os
TTS_ENDPOINT = _os.getenv("TTS_ENDPOINT", "")


# ── Default durations ──────────────────────────────────────────────────────────

POMODORO_DEFAULTS = {
    "focus_minutes":           25,
    "short_break_minutes":      5,
    "long_break_minutes":      15,
    "blocks_before_long_break": 4,
}

FREE_DEFAULTS = {
    "focus_minutes":  60,
    "break_minutes":  10,
}

ATTENTION_DEFAULTS = {
    "inactivity_threshold_seconds": 90,
    "response_window_seconds":      15,
}

SUMMARIES_DIR = "summaries"


# ── Session states ─────────────────────────────────────────────────────────────

IDLE      = "idle"
FOCUSING  = "focusing"
BLOCK_END = "block_end"
ON_BREAK  = "on_break"
PAUSED    = "paused"
ENDED     = "ended"


# ── Focus Session ──────────────────────────────────────────────────────────────

class FocusSession:
    """
    Runs a focus session in Pomodoro or Free mode.

    Integrates with CameraAttentionMonitor or AttentionMonitor depending
    on the user's accessibility profile. Saves a JSON summary at the end
    and reads it aloud in Egyptian Arabic.
    """

    def __init__(
        self,
        mode: str                                      = "pomodoro",
        focus_minutes: Optional[int]                   = None,
        break_minutes: Optional[int]                   = None,
        accessibility_profile: str                     = "standard",
        speak_fn: Optional[Callable[[str], None]]      = None,
        camera_source                                   = 0,
        frame_source: Optional[Callable]               = None,
    ):
        """
        Parameters
        ----------
        mode : str
            "pomodoro" or "free"
        focus_minutes : int, optional
            Override default focus block duration.
        break_minutes : int, optional
            Override default break duration.
        accessibility_profile : str
            "standard" | "low_vision" | "blind"
            Loaded from user_profile.json by SessionSetup — passed in here.
        speak_fn : callable(str)
            TTS function. If None, falls back to requests POST to TTS_ENDPOINT.
        camera_source : int or str
            Camera index or DroidCam URL string.
        frame_source : callable, optional
            External frame provider — overrides camera_source.
        """
        assert mode in ("pomodoro", "free"), "mode must be 'pomodoro' or 'free'"
        self.mode = mode

        # ── Accessibility profile ──────────────────────────────────────────────
        self._profile      = get_profile(accessibility_profile)
        self._profile_name = accessibility_profile
        logger.info("Accessibility profile: %s", accessibility_profile)

        # ── Durations ──────────────────────────────────────────────────────────
        self._pomodoro = dict(POMODORO_DEFAULTS)
        self._free      = dict(FREE_DEFAULTS)

        if focus_minutes:
            if mode == "pomodoro":
                self._pomodoro["focus_minutes"] = focus_minutes
            else:
                self._free["focus_minutes"] = focus_minutes

        if break_minutes:
            if mode == "pomodoro":
                self._pomodoro["short_break_minutes"] = break_minutes
            else:
                self._free["break_minutes"] = break_minutes

        # ── TTS ────────────────────────────────────────────────────────────────
        self._speak_fn = speak_fn or self._default_speak

        # ── STT — Azure, owned and managed internally by FocusSession ─────────
        # Started in start(), stopped in _end_session()
        # Whisper handles STT for the rest of the app — Azure is only for sessions
        self._stt    = None
        self._router = None
        try:
            from stt_handler import STTHandler
            from voice_command_router import VoiceCommandRouter

            self._stt    = STTHandler()
            self._router = VoiceCommandRouter(
                session = self,
                stt_fn  = None,   # router uses callbacks, not polling
            )

            # Register callbacks — both monitor and router get every STT result
            self._stt.add_callback(self._on_stt_text)

            logger.info("Azure STT and VoiceCommandRouter initialized.")
        except Exception as e:
            logger.warning(
                "Could not initialize Azure STT: %s. "
                "Monitors will rely on key presses only.", e
            )

        # ── State ──────────────────────────────────────────────────────────────
        self.state           = IDLE
        self.remaining_time  = None   # seconds left in block when paused

        # ── Stats ──────────────────────────────────────────────────────────────
        self._blocks_completed  = 0
        self._breaks_taken      = 0
        self._distraction_count = 0
        self._total_focus_secs  = 0
        self._total_break_secs  = 0
        self._distraction_times = []
        self._started_at        = None
        self._block_start       = None
        self._break_start       = None

        # ── Timer ──────────────────────────────────────────────────────────────
        self._timer_stop   = threading.Event()
        self._timer_thread = None

        # ── Attention monitor ──────────────────────────────────────────────────
        self._monitor = self._build_monitor(
            camera_source = camera_source,
            frame_source  = frame_source,
        )

    # ── Monitor factory ────────────────────────────────────────────────────────

    def _build_monitor(self, camera_source, frame_source):
        """
        Build the right monitor based on accessibility profile.
        blind profile → always no-camera.
        standard/low_vision → camera, with auto-fallback if it fails.
        Monitors no longer need stt_fn — speech reaches them via
        _on_stt_text callback which calls register_interaction() directly.
        """
        profile = self._profile

        if not profile.camera_enabled:
            logger.info("Profile is blind — using no-camera monitor.")
            return self._build_no_camera_monitor()

        logger.info("Building camera monitor (sensitivity=%s, head_pose_weight=%.1f).",
                    profile.camera_sensitivity, profile.head_pose_weight)

        from camera_attention_monitor import CameraAttentionMonitor
        return CameraAttentionMonitor(
            on_distraction       = self._on_distraction,
            on_resume            = self._on_resume,
            speak_fn             = self._speak_fn,
            stt_fn               = None,
            inactivity_threshold = ATTENTION_DEFAULTS["inactivity_threshold_seconds"],
            response_window      = profile.checkin_response_window,
            key_presses_enabled  = profile.key_presses_enabled,
            head_pose_weight     = profile.head_pose_weight,
            sensitivity          = profile.camera_sensitivity,
            camera_source        = camera_source,
            frame_source         = frame_source,
        )

    def _build_no_camera_monitor(self):
        from attention_monitor import AttentionMonitor
        profile = self._profile
        return AttentionMonitor(
            on_distraction       = self._on_distraction,
            on_resume            = self._on_resume,
            speak_fn             = self._speak_fn,
            stt_fn               = None,
            inactivity_threshold = ATTENTION_DEFAULTS["inactivity_threshold_seconds"],
            response_window      = profile.checkin_response_window,
            key_presses_enabled  = profile.key_presses_enabled,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Start the session, Azure STT, and voice command router."""
        self._started_at = datetime.now().isoformat()
        if self._stt:
            self._stt.start()
        if self._router:
            self._router.start()
        self._speak(PROMPTS["session_start"])
        self._start_focus_block()

    def _on_stt_text(self, text: str):
        """
        Callback fired by STTHandler for every recognized speech result.
        Routes to both the monitor (presence detection) and the router
        (command detection) so neither consumes the other's result.
        """
        # 1. Always register as interaction — proves user is present
        self._monitor.register_interaction()
        # 2. Let the router decide if it's a command
        if self._router:
            self._router.handle(text)

    def register_interaction(self):
        """Call on any user key press or button tap."""
        self._monitor.register_interaction()

    def user_choice(self, text: str):
        """
        Called with the user's spoken or typed block-end choice.
        Accepts Egyptian Arabic: "استراحة", "كمل", "خلاص"
        Also accepts numeric: "1", "2", "3"
        Parsed via accessibility_profile.parse_block_end_choice().
        """
        if self.state != BLOCK_END:
            return

        choice = parse_block_end_choice(text)

        if choice == "break":
            self._start_break()
        elif choice == "continue":
            self._start_focus_block()
        elif choice == "end":
            self._end_session()
        else:
            self._speak(PROMPTS["didnt_catch"])
            self._speak(PROMPTS["block_end"])

    def stop(self):
        """Force stop from outside (e.g. app close)."""
        self._end_session(forced=True)

    def switch_to_no_camera(self):
        """
        Called when user turns camera off mid-session.
        Transparently swaps monitor without interrupting the session.
        """
        logger.info("Camera disabled mid-session. Switching to no-camera monitor.")
        self._speak(PROMPTS["camera_disabled"])

        was_running = self.state in (FOCUSING, ON_BREAK, PAUSED)

        if was_running:
            self._monitor.stop()

        self._monitor = self._build_no_camera_monitor()

        if was_running:
            self._monitor.start()

    # ── State transitions ──────────────────────────────────────────────────────

    def _start_focus_block(self):
        self._set_state(FOCUSING)
        self._block_start   = time.time()
        self.remaining_time = None
        self._speak(PROMPTS["block_start"])

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
        self._speak(PROMPTS["block_end"])

    def _start_break(self):
        self._set_state(ON_BREAK)
        self._break_start = time.time()
        self._breaks_taken += 1
        self._speak(PROMPTS["break_start"])
        self._start_timer(self._break_duration(), self._on_break_end)

    def _on_break_end(self):
        if self.state != ON_BREAK:
            return

        self._total_break_secs += int(time.time() - self._break_start)
        self._speak(PROMPTS["break_end"])
        self._start_focus_block()

    def _end_session(self, forced: bool = False):
        self._set_state(ENDED)
        self._stop_timer()
        self._monitor.stop()
        if self._router:
            self._router.stop()
        if self._stt:
            self._stt.stop()

        if not forced:
            self._speak(PROMPTS["session_end"])

        self._save_and_read_summary()
        self._set_state(IDLE)

    # ── Pause / Resume ─────────────────────────────────────────────────────────

    def _pause_session(self):
        """Pause the block timer. Called by _on_distraction."""
        if self._block_start is None:
            logger.warning("_pause_session called but _block_start is None.")
            return

        elapsed             = int(time.time() - self._block_start)
        total               = self._focus_duration()
        self.remaining_time = max(total - elapsed, 10)

        self._stop_timer()
        self._set_state(PAUSED)
        logger.info(
            "Session paused. Remaining time: %ds.",
            self.remaining_time
        )

    def _on_distraction(self):
        """Callback from monitor — distraction confirmed."""
        if self.state != FOCUSING:
            return
        self._distraction_count += 1
        self._distraction_times.append(datetime.now().isoformat())
        logger.info("Distraction #%d recorded.", self._distraction_count)
        self._pause_session()

    def _on_resume(self):
        """Callback from monitor — user returned."""
        if self.state != PAUSED:
            return

        if self.remaining_time is None:
            logger.warning("No remaining time found. Restarting block.")
            self._start_focus_block()
            return

        logger.info("Resuming block. Remaining: %ds.", self.remaining_time)
        self._block_start = time.time()
        self._start_timer(self.remaining_time, self._on_block_end)
        self._set_state(FOCUSING)

    # ── Summary ────────────────────────────────────────────────────────────────

    def _save_and_read_summary(self):
        os.makedirs(SUMMARIES_DIR, exist_ok=True)

        summary = {
            "mode":                   self.mode,
            "accessibility_profile":  self._profile_name,
            "started_at":             self._started_at,
            "ended_at":               datetime.now().isoformat(),
            "blocks_completed":       self._blocks_completed,
            "breaks_taken":           self._breaks_taken,
            "distraction_count":      self._distraction_count,
            "distraction_timestamps": self._distraction_times,
            "total_focus_minutes":    round(self._total_focus_secs / 60, 1),
            "total_break_minutes":    round(self._total_break_secs / 60, 1),
        }

        filename = (
            f"{SUMMARIES_DIR}/session_"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
        )
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info("Summary saved → %s", filename)
        except IOError as e:
            logger.error("Could not save summary: %s", e)

        self._speak(PROMPTS["summary_intro"])
        self._speak(self._summary_as_speech(summary))

    def _summary_as_speech(self, s: dict) -> str:
        """Build Egyptian Arabic session summary for TTS."""
        blocks = s["blocks_completed"]
        focus  = s["total_focus_minutes"]
        breaks = s["breaks_taken"]
        dist   = s["distraction_count"]

        lines = [
            f"اتممت {blocks} {'بلوك تركيز' if blocks == 1 else 'بلوكات تركيز'}.",
            f"مجموع وقت التركيز كان {focus} دقيقة.",
            f"اخدت {breaks} {'استراحة' if breaks == 1 else 'استراحات'}.",
        ]

        if dist == 0:
            lines.append("مفيش تشتيت خالص. تركيز ممتاز!")
        elif dist == 1:
            lines.append("في تشتيت واحد اتسجل.")
        else:
            lines.append(f"اتسجل {dist} تشتيتات.")

        return " ".join(lines)

    # ── Timer helpers ──────────────────────────────────────────────────────────

    def _start_timer(self, seconds: int, callback: Callable):
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
            # Don't join if we're already on the timer thread —
            # happens when a callback (e.g. _on_break_end) starts a new block
            if threading.current_thread() is not self._timer_thread:
                self._timer_thread.join(timeout=2)

    def _focus_duration(self) -> int:
        if self.mode == "pomodoro":
            return self._pomodoro["focus_minutes"] * 60
        return self._free["focus_minutes"] * 60

    def _break_duration(self) -> int:
        if self.mode == "pomodoro":
            if self._blocks_completed % self._pomodoro["blocks_before_long_break"] == 0:
                return self._pomodoro["long_break_minutes"] * 60
            return self._pomodoro["short_break_minutes"] * 60
        return self._free["break_minutes"] * 60

    def _set_state(self, new_state: str):
        logger.info("State: %s → %s", self.state, new_state)
        self.state = new_state

    # ── TTS ────────────────────────────────────────────────────────────────────

    def _speak(self, text: str):
        try:
            self._speak_fn(text)
        except Exception as e:
            logger.error("TTS error: %s", e)

    def _default_speak(self, text: str):
        """Fallback TTS — POST to TTS_ENDPOINT if set, else log."""
        logger.info("[TTS] %s", text)
        if not TTS_ENDPOINT:
            return
        try:
            import requests
            requests.post(TTS_ENDPOINT, json={"text": text}, timeout=10)
        except Exception as e:
            logger.error("TTS endpoint error: %s", e)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def speak(text):
        print(f"[TTS] {text}")

    session = FocusSession(
        mode                  = "pomodoro",
        focus_minutes         = 1,
        break_minutes         = 1,
        accessibility_profile = "standard",
        speak_fn              = speak,
    )
    session.start()

    try:
        while True:
            key = input("Enter=interact | b=break | c=continue | e=end: ").strip()
            if key == "b":
                session.user_choice("استراحة")
            elif key == "c":
                session.user_choice("كمل")
            elif key == "e":
                session.user_choice("خلاص")
            else:
                session.register_interaction()
    except KeyboardInterrupt:
        session.stop()