"""
session_setup.py
────────────────
Collects session settings from the user through natural voice conversation
before handing off to FocusSession.

Handles two flows:

  First-time user:
    1. Welcome message
    2. Accessibility question → sets profile
    3. Mode question (pomodoro / free)
    4. Focus duration question
    5. Break duration question
    6. Confirmation → calls on_ready(settings)

  Returning user (user_profile.json exists):
    1. "هنبدأ زي المعتاد؟"
    2. If yes  → calls on_ready(saved settings) immediately
    3. If no   → runs full setup and saves new profile

Changes from v1:
  - Loads/saves user_profile.json via accessibility_profile helpers
  - Accessibility question added before session questions
  - All prompts from accessibility_profile.PROMPTS in Egyptian Arabic
  - Egyptian Arabic number words expanded
  - on_ready() now passes accessibility_profile in settings dict
  - logging throughout instead of print

Usage in app.py:

    setup = SessionSetup(
        on_ready  = lambda s: start_session(s),
        speak_fn  = tts.speak,
        initial_entities = entities,   # from brain API, optional
    )
    setup.begin()

    # On every incoming STT result while setup is pending:
    if setup.is_waiting:
        setup.fill_slot(user_text)
"""

import re
import logging
from typing import Callable, Optional

from accessibility_profile import (
    PROMPTS,
    load_user_profile,
    save_user_profile,
    parse_confirmation,
)

logger = logging.getLogger(__name__)


# ── Pomodoro defaults ──────────────────────────────────────────────────────────

POMODORO_DEFAULTS = {
    "focus_mins": 25,
    "break_mins":  5,
}

# ── Egyptian Arabic number word map ───────────────────────────────────────────
# Extended to cover common spoken Egyptian dialect forms

NUMBER_WORDS = {
    # English
    "five": 5,       "ten": 10,      "fifteen": 15,
    "twenty": 20,    "thirty": 30,   "forty": 40,
    "forty five": 45, "sixty": 60,   "ninety": 90,
    # Arabic — standard
    "خمسة": 5,       "عشرة": 10,     "خمسة عشر": 15,
    "عشرين": 20,     "ثلاثين": 30,   "أربعين": 40,
    "خمسة وأربعين": 45, "ستين": 60,  "تسعين": 90,
    # Egyptian dialect variants
    "خمستاشر": 15,   "عشرين": 20,    "تلاتين": 30,
    "أربعين": 40,    "خمسة وأربعين": 45,
    "ساعة": 60,      "ساعه": 60,     # "an hour"
}

# ── Accessibility keyword map ──────────────────────────────────────────────────

ACCESSIBILITY_KEYWORDS = {
    "blind": [
        "مكفوف", "مش شايف", "مش بشوف", "عمى", "قارئ شاشة",
        "blind", "screen reader",
    ],
    "low_vision": [
        "ضعف نظر", "بشوف بصعوبة", "نظري ضعيف", "مش شايف كويس",
        "low vision", "weak eyes",
    ],
}


class SessionSetup:
    """
    Guides the user through session configuration via voice conversation.
    Handles first-time and returning users differently.
    Saves profile to user_profile.json after first setup.
    """

    def __init__(
        self,
        on_ready: Callable,
        speak_fn: Callable[[str], None],
        initial_entities: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        on_ready : callable(settings: dict)
            Called when all slots are filled. Receives:
            {
                "mode":                   "pomodoro" | "free",
                "focus_mins":             int,
                "break_mins":             int,
                "accessibility_profile":  "standard" | "low_vision" | "blind",
            }
        speak_fn : callable(str)
            TTS function — called with Egyptian Arabic text.
        initial_entities : dict, optional
            Pre-extracted entities from the brain API.
            Keys: mode, focus_mins, duration_mins, break_mins
        """
        self._on_ready = on_ready
        self._speak    = speak_fn

        # ── Load saved profile ─────────────────────────────────────────────────
        self._user_profile  = load_user_profile()
        self._is_first_time = self._user_profile.get("first_time", True)

        # ── Slots ──────────────────────────────────────────────────────────────
        self._slots = {
            "accessibility_profile": None,
            "mode":                  None,
            "focus_mins":            None,
            "break_mins":            None,
        }

        # Pre-fill from brain API entities if provided
        if initial_entities:
            self._prefill(initial_entities)

        # ── Flow state ─────────────────────────────────────────────────────────
        self._pending_slot          = None
        self._awaiting_same_as_usual = False   # True when asking returning user

    # ── Public API ─────────────────────────────────────────────────────────────

    def begin(self):
        """
        Start setup conversation.
        Returning users get "هنبدأ زي المعتاد؟" first.
        First-time users get the full flow.
        """
        if self._is_first_time:
            self._speak(PROMPTS["first_time_welcome"])
            self._ask_next()
        else:
            self._speak(PROMPTS["welcome_back"])
            self._awaiting_same_as_usual = True

    def fill_slot(self, user_text: str):
        """
        Called with every incoming STT result while setup is in progress.
        Handles both the "same as usual?" confirmation and individual slot answers.
        """
        if not user_text:
            return

        # ── Returning user — "same as usual?" ─────────────────────────────────
        if self._awaiting_same_as_usual:
            self._handle_same_as_usual(user_text)
            return

        # ── Normal slot filling ────────────────────────────────────────────────
        if not self._pending_slot:
            return

        value = self._parse(self._pending_slot, user_text)

        if value is None:
            self._speak(PROMPTS["didnt_catch"])
            self._speak(self._question_for(self._pending_slot))
            return

        self._slots[self._pending_slot] = value
        logger.debug("Slot filled: %s = %s", self._pending_slot, value)
        self._pending_slot = None
        self._ask_next()

    @property
    def is_complete(self) -> bool:
        return all(v is not None for v in self._slots.values())

    @property
    def is_waiting(self) -> bool:
        """True while waiting for a user answer."""
        return self._pending_slot is not None or self._awaiting_same_as_usual

    # ── Returning user flow ────────────────────────────────────────────────────

    def _handle_same_as_usual(self, text: str):
        self._awaiting_same_as_usual = False

        if parse_confirmation(text):
            # User said yes — use saved profile directly
            logger.info("Returning user confirmed same settings.")
            settings = {
                "mode":                  self._user_profile["preferred_mode"],
                "focus_mins":            self._user_profile["focus_mins"],
                "break_mins":            self._user_profile["break_mins"],
                "accessibility_profile": self._user_profile["accessibility_profile"],
            }
            self._speak(
                PROMPTS["confirm_session"].format(
                    mode       = self._mode_display(settings["mode"]),
                    focus_mins = settings["focus_mins"],
                    break_mins = settings["break_mins"],
                )
            )
            self._on_ready(settings)
        else:
            # User wants to change something — run full setup
            logger.info("Returning user wants to change settings.")
            self._speak(PROMPTS["change_settings_question"])
            self._ask_next()

    # ── Slot flow ──────────────────────────────────────────────────────────────

    def _ask_next(self):
        """Ask the next unfilled slot, or confirm and fire on_ready if all done."""
        for slot, value in self._slots.items():
            if value is None:
                self._pending_slot = slot
                self._speak(self._question_for(slot))
                return

        # All slots filled
        self._confirm_and_start()

    def _confirm_and_start(self):
        """Speak confirmation, save profile, fire on_ready."""
        self._speak(
            PROMPTS["confirm_session"].format(
                mode       = self._mode_display(self._slots["mode"]),
                focus_mins = self._slots["focus_mins"],
                break_mins = self._slots["break_mins"],
            )
        )

        # Save to user_profile.json
        self._user_profile.update({
            "accessibility_profile": self._slots["accessibility_profile"],
            "preferred_mode":        self._slots["mode"],
            "focus_mins":            self._slots["focus_mins"],
            "break_mins":            self._slots["break_mins"],
            "camera_enabled":        self._slots["accessibility_profile"] != "blind",
            "first_time":            False,
        })
        save_user_profile(self._user_profile)
        logger.info("User profile saved.")

        self._on_ready(dict(self._slots))

    # ── Pre-fill from brain API ────────────────────────────────────────────────

    def _prefill(self, entities: dict):
        """Fill slots from brain API extracted entities."""
        mode = entities.get("mode", "").lower()
        if mode in ("pomodoro", "free"):
            self._slots["mode"] = mode

        focus = entities.get("focus_mins") or entities.get("duration_mins")
        if focus:
            self._slots["focus_mins"] = int(focus)

        brk = entities.get("break_mins")
        if brk:
            self._slots["break_mins"] = int(brk)

        # Pomodoro with no times → apply defaults silently
        if self._slots["mode"] == "pomodoro":
            if not self._slots["focus_mins"]:
                self._slots["focus_mins"] = POMODORO_DEFAULTS["focus_mins"]
            if not self._slots["break_mins"]:
                self._slots["break_mins"] = POMODORO_DEFAULTS["break_mins"]

        logger.debug("Pre-filled slots from entities: %s", self._slots)

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse(self, slot: str, text: str):
        """Parse STT text for a given slot. Returns value or None."""
        text = text.strip()
        text_lower = text.lower()

        if slot == "accessibility_profile":
            return self._parse_accessibility(text_lower)

        if slot == "mode":
            return self._parse_mode(text_lower)

        if slot in ("focus_mins", "break_mins"):
            return self._parse_duration(text_lower)

        return None

    def _parse_accessibility(self, text: str) -> Optional[str]:
        for profile, keywords in ACCESSIBILITY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                logger.info("Accessibility profile detected: %s", profile)
                if profile == "blind":
                    self._speak(PROMPTS["accessibility_confirm_blind"])
                else:
                    self._speak(PROMPTS["accessibility_confirm_low_vision"])
                return profile

        # Any confirmation or "no" / "normal" → standard
        if any(w in text for w in ["لا", "عادي", "كويس", "مفيش", "no", "normal", "fine"]):
            self._speak(PROMPTS["accessibility_confirm_standard"])
            return "standard"

        # Anything else (e.g. user just says "يلا") → standard
        if text:
            self._speak(PROMPTS["accessibility_confirm_standard"])
            return "standard"

        return None

    def _parse_mode(self, text: str) -> Optional[str]:
        if any(w in text for w in ("pomodoro", "بومودورو", "بوميدورو")):
            if not self._slots["focus_mins"]:
                self._slots["focus_mins"] = POMODORO_DEFAULTS["focus_mins"]
            if not self._slots["break_mins"]:
                self._slots["break_mins"] = POMODORO_DEFAULTS["break_mins"]
            return "pomodoro"

        if any(w in text for w in ("free", "حر", "حرة", "مفتوح", "open")):
            return "free"

        return None

    def _parse_duration(self, text: str) -> Optional[int]:
        # Digit extraction first
        match = re.search(r"\d+", text)
        if match:
            return int(match.group())

        # Spoken word map
        for word, value in NUMBER_WORDS.items():
            if word in text:
                return value

        return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _question_for(self, slot: str) -> str:
        """Return the Egyptian Arabic question for a given slot."""
        return {
            "accessibility_profile": PROMPTS["accessibility_question"],
            "mode":                  PROMPTS["ask_mode"],
            "focus_mins":            PROMPTS["ask_focus_mins"],
            "break_mins":            PROMPTS["ask_break_mins"],
        }.get(slot, PROMPTS["didnt_catch"])

    def _mode_display(self, mode: str) -> str:
        """Egyptian Arabic display name for mode."""
        return "بومودورو" if mode == "pomodoro" else "حرة"