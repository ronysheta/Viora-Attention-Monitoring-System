"""
attention_monitor.py
────────────────────
No-camera attention monitor for Viora.

Tracks user inactivity and voice activity during a study session.
When the user goes silent for `inactivity_threshold` seconds, a
voice check-in is played via TTS. The user can respond by:
  - Speaking anything (detected via STT)
  - Pressing any key (if key_presses_enabled in their profile)

If no response is received within `response_window` seconds,
the session is paused and `on_distraction` is fired.
When the user responds after a pause, `on_resume` is fired.

All prompts are in Egyptian Arabic dialect, sourced from
accessibility_profile.py.

Changes from v1:
  - Added `on_resume` callback (was missing entirely)
  - Added `stt_fn` — any detected speech auto-calls register_interaction()
  - Added `key_presses_enabled` — if False, never mention pressing anything
  - Check-in response now detected via STT as well as key press
  - All messages sourced from accessibility_profile.PROMPTS
  - Added STT polling loop running alongside the inactivity loop
  - Added paused state so resume is handled cleanly

Usage:
    from accessibility_profile import PROMPTS
    from attention_monitor import AttentionMonitor

    monitor = AttentionMonitor(
        on_distraction = session.on_distraction,
        on_resume      = session.on_resume,
        speak_fn       = tts.speak,
        stt_fn         = stt.get_latest_text,   # returns str or None
        key_presses_enabled = True,
    )
    monitor.start()
    monitor.register_interaction()   # call on any key press
    monitor.stop()
"""

import time
import threading
from typing import Callable, Optional

from accessibility_profile import PROMPTS, parse_confirmation


class AttentionMonitor:

    def __init__(
        self,
        on_distraction: Optional[Callable]        = None,
        on_resume: Optional[Callable]             = None,
        speak_fn: Optional[Callable[[str], None]] = None,
        stt_fn: Optional[Callable[[], Optional[str]]] = None,
        inactivity_threshold: int                  = 90,
        response_window: int                       = 15,
        key_presses_enabled: bool                  = True,
    ):
        """
        Parameters
        ----------
        on_distraction : callable
            Fired when distraction is confirmed — use to pause session.
        on_resume : callable
            Fired when user responds after a pause — use to resume session.
        speak_fn : callable(str)
            TTS function — called with Egyptian Arabic text.
        stt_fn : callable() -> str | None
            Returns the latest STT-detected text, or None if nothing heard.
            Called every 0.5s by the STT polling loop.
            Any non-None return auto-calls register_interaction().
        inactivity_threshold : int
            Seconds of silence before a check-in is triggered.
        response_window : int
            Seconds to wait for a response after check-in before
            confirming distraction.
        key_presses_enabled : bool
            If False, check-in message never mentions pressing anything.
            Set automatically from AccessibilityProfile.
        """
        self.on_distraction       = on_distraction
        self.on_resume            = on_resume
        self.inactivity_threshold = inactivity_threshold
        self.response_window      = response_window
        self.key_presses_enabled  = key_presses_enabled

        self._speak_fn = speak_fn or (lambda t: print(f"[AttentionMonitor] TTS: {t}"))
        self._stt_fn   = stt_fn   # None = no STT, rely on key presses only

        # ── State ──────────────────────────────────────────────────────────────
        self._last_interaction      = time.time()
        self._running               = False
        self._paused                = False
        self._waiting_for_response  = False
        self._response_received     = False

        # ── Threads ────────────────────────────────────────────────────────────
        self._monitor_thread        = None
        self._stt_thread            = None
        self._lock                  = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Start background monitoring and STT polling threads."""
        self._running          = True
        self._paused           = False
        self._last_interaction = time.time()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()

        # Only start STT thread if an stt_fn was provided
        if self._stt_fn:
            self._stt_thread = threading.Thread(
                target=self._stt_loop, daemon=True
            )
            self._stt_thread.start()

        print("[AttentionMonitor] Started.")

    def stop(self):
        """Stop all background threads cleanly."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        if self._stt_thread:
            self._stt_thread.join(timeout=2)
        print("[AttentionMonitor] Stopped.")

    def register_interaction(self):
        """
        Call this on any user interaction: key press, button tap, etc.
        Also called automatically by the STT loop when speech is detected.
        Resets the inactivity timer and cancels any pending check-in.
        If session was paused, fires on_resume.
        """
        was_paused = False

        with self._lock:
            self._last_interaction = time.time()

            if self._waiting_for_response:
                self._response_received  = True
                self._waiting_for_response = False
                print("[AttentionMonitor] User responded to check-in. Timer reset.")

            if self._paused:
                self._paused   = False
                was_paused     = True

        if was_paused:
            print("[AttentionMonitor] Session resumed.")
            self._speak(PROMPTS["resume_prompt"])
            if self.on_resume:
                self.on_resume()

    def set_threshold(self, seconds: int):
        """Adjust inactivity threshold at runtime."""
        self.inactivity_threshold = seconds

    # ── STT polling loop ───────────────────────────────────────────────────────

    def _stt_loop(self):
        """
        Polls stt_fn every 0.5 seconds.
        Any detected speech automatically registers as an interaction.
        During a check-in wait, a spoken confirmation also counts as a response.
        """
        while self._running:
            time.sleep(0.5)

            if not self._stt_fn:
                continue

            try:
                text = self._stt_fn()
            except Exception as e:
                print(f"[AttentionMonitor] STT error: {e}")
                continue

            if text:
                print(f"[AttentionMonitor] STT detected: '{text}'")
                self.register_interaction()

    # ── Internal monitoring loop ───────────────────────────────────────────────

    def _monitor_loop(self):
        while self._running:
            time.sleep(1)

            with self._lock:
                elapsed          = time.time() - self._last_interaction
                already_waiting  = self._waiting_for_response
                paused           = self._paused

            # Don't re-trigger while paused or check-in already in progress
            if already_waiting or paused:
                continue

            if elapsed >= self.inactivity_threshold:
                self._trigger_checkin()

    def _trigger_checkin(self):
        """Play check-in prompt and wait for voice or key response."""
        print("[AttentionMonitor] Inactivity threshold reached. Playing check-in.")

        with self._lock:
            self._waiting_for_response = True
            self._response_received    = False

        # Pick the right check-in message based on key_presses_enabled
        if self.key_presses_enabled:
            checkin_msg = PROMPTS["checkin_no_camera"]
        else:
            checkin_msg = PROMPTS["checkin_blind"]

        self._speak(checkin_msg)

        # Wait for response within the response window
        deadline = time.time() + self.response_window
        while time.time() < deadline:
            time.sleep(0.5)
            with self._lock:
                if self._response_received:
                    self._last_interaction = time.time()
                    print("[AttentionMonitor] Response received in time.")
                    return

        # No response — confirm distraction
        with self._lock:
            self._waiting_for_response = False
            self._paused               = True

        print("[AttentionMonitor] No response. Distraction confirmed.")
        self._trigger_distraction()

    def _trigger_distraction(self):
        """Fire distraction alert and callback."""
        self._speak(PROMPTS["distraction_confirmed"])

        with self._lock:
            self._last_interaction = time.time()

        if self.on_distraction:
            self.on_distraction()

    def _speak(self, text: str):
        try:
            self._speak_fn(text)
        except Exception as e:
            print(f"[AttentionMonitor] TTS error: {e}")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== Attention Monitor Test ===")
    print("Inactivity threshold: 8 seconds")
    print("Response window: 5 seconds")
    print("Press Enter to simulate interaction. Ctrl+C to stop.\n")

    def speak(text):
        print(f"[TTS] {text}")

    def on_distraction_detected():
        print("\n>>> SESSION PAUSED\n")

    def on_resume_detected():
        print("\n>>> SESSION RESUMED\n")

    # Simulate STT — returns "أيوه" once after 3 seconds then nothing
    _stt_trigger = {"fired": False, "at": time.time() + 20}
    def fake_stt():
        if not _stt_trigger["fired"] and time.time() > _stt_trigger["at"]:
            _stt_trigger["fired"] = True
            return "أيوه"
        return None

    monitor = AttentionMonitor(
        on_distraction      = on_distraction_detected,
        on_resume           = on_resume_detected,
        speak_fn            = speak,
        stt_fn              = fake_stt,
        inactivity_threshold = 8,
        response_window      = 5,
        key_presses_enabled  = True,
    )

    monitor.start()

    try:
        while True:
            input()
            monitor.register_interaction()
            print("[Test] Interaction registered.")
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop()
        sys.exit(0)