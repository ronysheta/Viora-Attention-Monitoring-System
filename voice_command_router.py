"""
voice_command_router.py
───────────────────────
Routes Azure STT output to the correct FocusSession action.

Receives speech text via handle() which is called directly by
FocusSession._on_stt_text() every time Azure recognizes something.
No polling needed — the callback system in STTHandler delivers
every result to both the monitor and the router simultaneously.

Routing logic:
  "وقف" / "خلاص"            → session.stop()
  "استراحة" / "خد استراحة"  → session.user_choice("استراحة")  [only at block_end]
  "كمل" / "استمر"           → session.user_choice("كمل")      [only at block_end]
  anything else              → ignored (monitor already called register_interaction)

Usage:
    router = VoiceCommandRouter(session=focus_session)
    router.start()
    router.handle("وقف")   # called automatically by FocusSession._on_stt_text
    router.stop()
"""

import logging
from typing import Optional

from accessibility_profile import (
    STOP_WORDS,
    parse_block_end_choice,
)

logger = logging.getLogger(__name__)


class VoiceCommandRouter:

    def __init__(self, session, stt_fn=None):
        """
        Parameters
        ----------
        session : FocusSession
            The active session to route commands to.
        stt_fn : ignored
            Kept for API compatibility — routing is now callback-based.
        """
        self._session = session
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        logger.info("VoiceCommandRouter started.")

    def stop(self):
        self._running = False
        logger.info("VoiceCommandRouter stopped.")

    def handle(self, text: str):
        """
        Called by FocusSession._on_stt_text() with every STT result.
        Decides if the text is a command and routes it accordingly.
        """
        if not self._running or not text:
            return

        text_lower = text.strip().lower()
        logger.debug("Routing: '%s' | state: %s", text, self._session.state)

        # ── Stop / end session ─────────────────────────────────────────────────
        if any(w in text_lower for w in STOP_WORDS):
            logger.info("Voice command: STOP")
            self._session.stop()
            return

        # ── Block end choices — only valid when state is block_end ─────────────
        if self._session.state == "block_end":
            choice = parse_block_end_choice(text_lower)
            if choice:
                logger.info("Voice command: block_end → %s", choice)
                self._session.user_choice(text)
                return

        # ── Anything else is just presence — monitor already handled it ────────
        logger.debug("No command matched — presence already registered by monitor.")
