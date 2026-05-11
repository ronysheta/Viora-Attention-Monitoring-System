import re


# Questions Viora asks for each missing slot
QUESTIONS = {
    "mode":        "Would you like a Pomodoro session or a free session?",
    "focus_mins":  "How many minutes would you like to focus?",
    "break_mins":  "How many minutes for each break?",
}

# Confirmation before starting
CONFIRM_TEMPLATE = (
    "Great! Starting a {mode} session with {focus_mins} minute focus blocks "
    "and {break_mins} minute breaks. Let's go!"
)

# Pomodoro defaults — used when user picks pomodoro but doesn't specify times
POMODORO_DEFAULTS = {
    "focus_mins": 25,
    "break_mins": 5,
}


class SessionSetup:
    """
    Collects mode, focus duration, and break length through voice conversation.

    If the user's initial command already contains some or all of this info,
    those slots are pre-filled and the corresponding questions are skipped.

    Once all slots are filled, on_ready(settings) is called with a dict:
        { "mode": "pomodoro"|"free", "focus_mins": int, "break_mins": int }

    Usage in app.py:

        # When brain API returns EXECUTE_SESSION_START:
        setup = SessionSetup(
            on_ready=lambda s: start_session(s),
            speak_fn=lambda t: socketio.emit('server_response', {'text': t}),
            initial_entities=entities,   # from brain API response
        )
        setup.begin()

        # On the next user voice command (while setup is pending):
        if active_setup:
            active_setup.fill_slot(user_text)
    """

    def __init__(self, on_ready, speak_fn, initial_entities: dict = None):
        self._on_ready = on_ready
        self._speak   = speak_fn

        # Slots — None means not yet filled
        self._slots = {
            "mode":       None,
            "focus_mins": None,
            "break_mins": None,
        }

        # Pre-fill from brain API entities if already extracted
        if initial_entities:
            self._prefill(initial_entities)

        self._pending_slot = None   # which slot we're currently waiting for

    # ── Public API ────────────────────────────────────────────────────────────

    def begin(self):
        """Start the setup — ask first missing question or start immediately."""
        self._ask_next()

    def fill_slot(self, user_text: str):
        """
        Called with the user's voice response while setup is in progress.
        Parses the answer, fills the pending slot, then asks the next one.
        """
        if not self._pending_slot:
            return

        value = self._parse(self._pending_slot, user_text)

        if value is None:
            self._speak(f"Sorry, I didn't catch that. {QUESTIONS[self._pending_slot]}")
            return

        self._slots[self._pending_slot] = value
        self._pending_slot = None
        self._ask_next()

    @property
    def is_complete(self) -> bool:
        return all(v is not None for v in self._slots.values())

    @property
    def is_waiting(self) -> bool:
        """True while we are mid-conversation waiting for a user answer."""
        return self._pending_slot is not None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prefill(self, entities: dict):
        """Fill slots from whatever the brain API already extracted."""
        mode = entities.get("mode")
        if mode and mode.lower() in ("pomodoro", "free"):
            self._slots["mode"] = mode.lower()

        focus = entities.get("focus_mins") or entities.get("duration_mins")
        if focus:
            self._slots["focus_mins"] = int(focus)

        brk = entities.get("break_mins")
        if brk:
            self._slots["break_mins"] = int(brk)

        # If user chose pomodoro and gave no times, use defaults silently
        if self._slots["mode"] == "pomodoro":
            if not self._slots["focus_mins"]:
                self._slots["focus_mins"] = POMODORO_DEFAULTS["focus_mins"]
            if not self._slots["break_mins"]:
                self._slots["break_mins"] = POMODORO_DEFAULTS["break_mins"]

    def _ask_next(self):
        """Ask the next unfilled slot, or confirm and start if all filled."""
        for slot, value in self._slots.items():
            if value is None:
                self._pending_slot = slot
                self._speak(QUESTIONS[slot])
                return

        # All slots filled — confirm and start
        confirm = CONFIRM_TEMPLATE.format(
            mode       = self._slots["mode"].capitalize(),
            focus_mins = self._slots["focus_mins"],
            break_mins = self._slots["break_mins"],
        )
        self._speak(confirm)
        self._on_ready(dict(self._slots))

    def _parse(self, slot: str, text: str):
        """Parse user's spoken answer for a given slot. Returns value or None."""
        text = text.lower().strip()

        if slot == "mode":
            if any(w in text for w in ("pomodoro", "بومودورو")):
                # Apply pomodoro defaults for unasked time slots
                if not self._slots["focus_mins"]:
                    self._slots["focus_mins"] = POMODORO_DEFAULTS["focus_mins"]
                if not self._slots["break_mins"]:
                    self._slots["break_mins"] = POMODORO_DEFAULTS["break_mins"]
                return "pomodoro"
            if any(w in text for w in ("free", "حر", "open")):
                return "free"
            return None

        if slot in ("focus_mins", "break_mins"):
            # Extract the first number mentioned
            match = re.search(r"\d+", text)
            if match:
                return int(match.group())
            # Handle spoken Arabic/English words
            word_map = {
                "five": 5, "ten": 10, "fifteen": 15, "twenty": 20,
                "thirty": 30, "forty": 40, "forty five": 45, "sixty": 60,
                "خمسة": 5, "عشرة": 10, "خمسة عشر": 15, "عشرين": 20,
                "ثلاثين": 30, "أربعين": 40, "خمسة وأربعين": 45, "ستين": 60,
            }
            for word, value in word_map.items():
                if word in text:
                    return value
            return None

        return None
