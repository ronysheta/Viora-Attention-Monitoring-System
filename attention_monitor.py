import time
import threading
from typing import Callable, Optional


class AttentionMonitor:
    """
    No-camera attention monitor.

    Tracks user inactivity during a study session. If the user goes
    silent for `inactivity_threshold` seconds, a voice check-in is played.
    If they don't respond within `response_window` seconds, the session
    is paused and `on_distraction` callback is fired.

    Usage:
        from tts_handler import TTSHandler
        tts = TTSHandler()

        monitor = AttentionMonitor(
            on_distraction=tts.pause,
            speak_fn=tts.speak,
        )
        monitor.start()

        # Call this whenever the user interacts with your app:
        monitor.register_interaction()

        monitor.stop()
    """

    def __init__(
        self,
        on_distraction: Optional[Callable] = None,
        speak_fn: Optional[Callable[[str], None]] = None,
        inactivity_threshold: int = 90,
        response_window: int = 15,
        checkin_message: str = "Are you still there? Press any key to continue.",
        alert_message: str = "It seems you may have drifted off. Let's take a short break and come back.",
    ):
        self.on_distraction = on_distraction
        self.inactivity_threshold = inactivity_threshold
        self.response_window = response_window
        self.checkin_message = checkin_message
        self.alert_message = alert_message

        # Use provided speak_fn or fall back to print
        self._speak_fn = speak_fn or (lambda text: print(f"[AttentionMonitor] TTS: {text}"))

        self._last_interaction = time.time()
        self._running = False
        self._waiting_for_response = False
        self._response_received = False
        self._thread = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Public API — call these from your app                               #
    # ------------------------------------------------------------------ #

    def start(self):
        """Start the background monitoring thread."""
        self._running = True
        self._last_interaction = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[AttentionMonitor] Started.")

    def stop(self):
        """Stop monitoring cleanly."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[AttentionMonitor] Stopped.")

    def register_interaction(self):
        """
        Call this from your app whenever the user does anything:
        key press, pause, rewind, speed change, etc.
        This resets the inactivity timer and cancels any pending check-in.
        """
        with self._lock:
            self._last_interaction = time.time()
            if self._waiting_for_response:
                # User responded to the check-in in time
                self._response_received = True
                self._waiting_for_response = False
                print("[AttentionMonitor] User responded. Timer reset.")

    def set_threshold(self, seconds):
        """Adjust inactivity threshold at runtime (e.g. from settings screen)."""
        self.inactivity_threshold = seconds

    # ------------------------------------------------------------------ #
    #  Internal monitoring loop                                            #
    # ------------------------------------------------------------------ #

    def _monitor_loop(self):
        while self._running:
            time.sleep(1)  # check every second

            with self._lock:
                elapsed = time.time() - self._last_interaction
                already_waiting = self._waiting_for_response

            if already_waiting:
                continue  # check-in already in progress, handled separately

            if elapsed >= self.inactivity_threshold:
                self._trigger_checkin()

    def _trigger_checkin(self):
        """Play check-in prompt and wait for response."""
        print("[AttentionMonitor] Inactivity threshold reached. Playing check-in.")

        with self._lock:
            self._waiting_for_response = True
            self._response_received = False

        # Play the check-in voice prompt
        self._speak(self.checkin_message)

        # Wait for response_window seconds
        deadline = time.time() + self.response_window
        while time.time() < deadline:
            time.sleep(0.5)
            with self._lock:
                if self._response_received:
                    # User responded in time — reset and resume
                    self._last_interaction = time.time()
                    return

        # No response received — confirm distraction
        with self._lock:
            self._waiting_for_response = False

        print("[AttentionMonitor] No response. Distraction confirmed.")
        self._trigger_distraction()

    def _trigger_distraction(self):
        """Fire the distraction alert and callback."""
        self._speak(self.alert_message)

        # Reset timer so we don't immediately re-trigger
        with self._lock:
            self._last_interaction = time.time()

        # Fire the callback (e.g. pause your TTS reading)
        if self.on_distraction:
            self.on_distraction()

    def _speak(self, text):
        """Speak a message via the injected speak function."""
        try:
            self._speak_fn(text)
        except Exception as e:
            print(f"[AttentionMonitor] TTS error: {e}")


# ------------------------------------------------------------------ #
#  Quick test — run this file directly to try it out                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys

    print("=== Attention Monitor Test ===")
    print("Inactivity threshold: 8 seconds (shortened for testing)")
    print("Response window: 5 seconds")
    print("Press Enter at any time to simulate a user interaction.\n")

    def speak(text):
        print(f"[TTS] {text}")

    def on_distraction_detected():
        print("\n>>> SESSION PAUSED — distraction confirmed.\n")

    monitor = AttentionMonitor(
        on_distraction=on_distraction_detected,
        speak_fn=speak,
        inactivity_threshold=8,
        response_window=5,
        checkin_message="Are you still there? Press Enter to continue.",
        alert_message="It looks like you drifted off. Pausing your session now.",
    )

    monitor.start()

    try:
        while True:
            input()
            monitor.register_interaction()
            print("[Test] Interaction registered. Timer reset.")
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()
        sys.exit(0)
