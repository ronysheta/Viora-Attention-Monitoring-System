"""
accessibility_profile.py
────────────────────────
Foundation file for Viora's accessibility system.

Defines:
  - Three accessibility profiles (standard, low_vision, blind)
  - All Egyptian Arabic spoken prompts in one place
  - User profile load/save to user_profile.json
  - Internal sensitivity mapping (never exposed to user)

Everything else imports from here. This file imports nothing from
the rest of the project — it is the clean base layer.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Profile definitions ────────────────────────────────────────────────────────

@dataclass
class AccessibilityProfile:
    """
    Defines how the session behaves for a given user type.
    Never instantiated directly — use PROFILES dict below.
    """
    name: str

    # Camera behaviour
    camera_enabled: bool        # whether to attempt camera-based monitoring
    head_pose_weight: float     # 0.0–1.0; lower = rely more on voice/interaction

    # Interaction behaviour
    stt_primary: bool           # if True, STT is the main interaction method
    key_presses_enabled: bool   # if False, never ask user to press anything

    # Internal camera sensitivity preset
    # "low" | "medium" | "high" — set automatically, never shown to user
    camera_sensitivity: str

    # How often to do voice check-ins (seconds) when suspicion is raised
    checkin_response_window: int


# ── The three profiles ─────────────────────────────────────────────────────────

PROFILES = {
    "standard": AccessibilityProfile(
        name                   = "standard",
        camera_enabled         = True,
        head_pose_weight       = 0.7,
        stt_primary            = False,
        key_presses_enabled    = True,
        camera_sensitivity     = "low",
        checkin_response_window = 15,
    ),

    "low_vision": AccessibilityProfile(
        name                   = "low_vision",
        camera_enabled         = True,       # camera still useful for head pose
        head_pose_weight       = 0.5,        # but we trust it less
        stt_primary            = True,       # voice is primary interaction
        key_presses_enabled    = False,      # never ask to press anything
        camera_sensitivity     = "low",
        checkin_response_window = 20,        # give more time to respond
    ),

    "blind": AccessibilityProfile(
        name                   = "blind",
        camera_enabled         = False,      # camera completely off
        head_pose_weight       = 0.0,        # irrelevant
        stt_primary            = True,
        key_presses_enabled    = False,
        camera_sensitivity     = "low",      # irrelevant but kept for consistency
        checkin_response_window = 25,        # most generous response window
    ),
}


# ── Egyptian Arabic prompts ────────────────────────────────────────────────────
# All spoken output in the app comes from here.
# Egyptian dialect throughout — never MSA (Modern Standard Arabic).

PROMPTS = {

    # ── Session setup ──────────────────────────────────────────────────────────
    "welcome_back":
        "أهلاً! هنبدأ جلسة زي المعتاد؟",
    "welcome_back_settings":
        "أهلاً! قولي لو عايز تغير أي حاجة، أو قول ابدأ ونبدأ على طول.",
    "first_time_welcome":
        "أهلاً بيك في فيورا! أنا هساعدك تركز وتذاكر بشكل أحسن.",
    "accessibility_question":
        "قبل ما نبدأ، هل تحتاج مساعدة خاصة؟ "
        "مثلاً لو عندك صعوبة في الرؤية أو بتستخدم قارئ شاشة قولي وهنضبطلك كل حاجة.",
    "accessibility_confirm_standard":
        "تمام! هنشتغل بالإعدادات العادية.",
    "accessibility_confirm_low_vision":
        "تمام! هنضبط كل حاجة عشان تكون أسهل عليك.",
    "accessibility_confirm_blind":
        "تمام! هنشتغل بالصوت بس وهتلاقي كل حاجة سهلة إن شاء الله.",
    "ask_mode":
        "عايز جلسة بومودورو ولا جلسة حرة؟",
    "ask_focus_mins":
        "كام دقيقة عايز تركز؟",
    "ask_break_mins":
        "وكام دقيقة للاستراحة؟",
    "confirm_session":
        "تمام! هنبدأ جلسة {mode} بتركيز {focus_mins} دقيقة واستراحة {break_mins} دقيقة. يلا بسم الله!",
    "change_settings_question":
        "إيه اللي عايز تغيره؟",
    "didnt_catch":
        "آسف معرفتش أفهم. ممكن تعيد؟",

    # ── Session flow ───────────────────────────────────────────────────────────
    "session_start":
        "جلستك بدأت. بالتوفيق!",
    "block_start":
        "بدأ وقت التركيز. هكلمك لما يخلص.",
    "block_end":
        "أحسنت! الجزء ده خلص. قول استراحة تاخد استراحة، "
        "أو كمّل تفضل شغال، أو خلاص تنهي الجلسة.",
    "break_start":
        "وقت الاستراحة. استرح شوية وهكلمك لما تخلص.",
    "break_end":
        "خلصت الاستراحة. قولي لما تبقى جاهز نكمل.",
    "session_end":
        "الجلسة خلصت. عملت حاجة كويسة النهارده!",
    "summary_intro":
        "خليني أقولك إيه اللي عملته النهارده.",

    # ── Attention check-ins ────────────────────────────────────────────────────
    "checkin_camera":
        "لسه معانا؟ قول أي حاجة نكمل.",
    "checkin_no_camera":
        "لسه شاغل بالك؟ قول أي حاجة نكمل.",
    "checkin_blind":
        "لسه معايا؟ قول أي حاجة وهنكمل على طول.",
    "distraction_confirmed":
        "يمكن اتشتت شوية. خليني أوقف الجلسة دلوقتي.",
    "resume_prompt":
        "أهلاً! رجعت تاني. هنكمل من حيث وقفنا.",
    "false_positive_offer":
        "لو الجلسة بتوقف كتير من غير ما تتشتت، قولي وهنضبطها.",

    # ── Comprehension check-ins ────────────────────────────────────────────────
    # Asked when distraction is suspected — confirms presence AND reinforces memory
    "comprehension_check_standard":
        "سؤال سريع — إيه آخر حاجة اتكلمنا فيها؟",
    "comprehension_check_blind":
        "سؤال سريع — إيه آخر حاجة سمعتها؟",

    # ── Camera status ──────────────────────────────────────────────────────────
    "camera_failed":
        "معرفتش أفتح الكاميرا. هنكمل من غير كاميرا عادي.",
    "camera_disabled":
        "تمام، الكاميرا اتوقفت. هنكمل بالصوت بس.",
    "calibrating":
        "استنى ثواني بنظبطلك الكاميرا.",
    "calibration_done":
        "تمام! كل حاجة اتظبطت.",
}


# ── User profile load / save ───────────────────────────────────────────────────

USER_PROFILE_PATH = "user_profile.json"

DEFAULT_USER_PROFILE = {
    "accessibility_profile": "standard",
    "preferred_mode":        "pomodoro",
    "focus_mins":            25,
    "break_mins":            5,
    "camera_enabled":        True,
    "first_time":            True,       # False after first setup
}


def load_user_profile() -> dict:
    """
    Load user profile from disk.
    Returns default profile if file doesn't exist yet (first time user).
    """
    if not os.path.exists(USER_PROFILE_PATH):
        return dict(DEFAULT_USER_PROFILE)

    try:
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)

        # Merge with defaults to handle missing keys from older versions
        profile = dict(DEFAULT_USER_PROFILE)
        profile.update(saved)
        return profile

    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Could not load profile: %s. Using defaults.", e)
        return dict(DEFAULT_USER_PROFILE)


def save_user_profile(profile: dict):
    """
    Save user profile to disk.
    Called after first setup and after any user-requested changes.
    """
    try:
        with open(USER_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        logger.info("Profile saved → %s", USER_PROFILE_PATH)
    except IOError as e:
        logger.error("Could not save profile: %s", e)


def get_profile(name: str) -> AccessibilityProfile:
    """
    Get an AccessibilityProfile by name.
    Falls back to standard if name is unrecognised.
    """
    if name not in PROFILES:
        logger.warning("Unknown profile '%s', using 'standard'.", name)
        return PROFILES["standard"]
    return PROFILES[name]


def get_checkin_message(profile_name: str) -> str:
    """
    Return the right check-in message for the given profile.
    """
    if profile_name == "blind":
        return PROMPTS["checkin_blind"]
    elif profile_name == "low_vision":
        return PROMPTS["checkin_no_camera"]
    else:
        return PROMPTS["checkin_camera"]


def get_comprehension_check(profile_name: str) -> str:
    """
    Return the right comprehension check message for the given profile.
    """
    if profile_name == "blind":
        return PROMPTS["comprehension_check_blind"]
    return PROMPTS["comprehension_check_standard"]


# ── STT response parsing ───────────────────────────────────────────────────────
# Keywords the STT might return that we treat as confirmations or commands.
# Egyptian dialect variants included throughout.

CONFIRMATION_WORDS = [
    "أيوه", "آه", "أه", "تمام", "اوكي", "اوك", "أكيد",
    "موجود", "هنا", "معاك", "مع حضرتك", "نعم", "يلا", "كمل",
    "yes", "yeah", "ok", "okay", "here", "continue",
]

CHANGE_SETTINGS_WORDS = [
    "غير", "عايز أغير", "بدل", "عدل", "إعدادات",
    "change", "settings", "modify",
]

START_WORDS = [
    "ابدأ", "يلا", "هيا", "نبدأ", "استمر", "كمل",
    "start", "begin", "go",
]

STOP_WORDS = [
    "وقف", "خلاص", "استنى", "بوقف", "إيقاف",
    "stop", "pause", "end",
]

BREAK_WORDS = [
    "استراحة", "راحة", "استريح",
    "break", "rest",
]

CONTINUE_WORDS = [
    "كمّل", "كمل", "استمر", "شغال", "خليني أكمل",
    "continue", "keep going",
]


def parse_confirmation(text: str) -> bool:
    """Returns True if the text contains any confirmation word."""
    text = text.lower().strip()
    return any(word in text for word in CONFIRMATION_WORDS)


def parse_block_end_choice(text: str) -> Optional[str]:
    """
    Parse user's spoken block-end choice.
    Returns "break", "continue", "end", or None if not recognised.
    """
    text = text.lower().strip()

    if any(w in text for w in BREAK_WORDS):
        return "break"
    if any(w in text for w in CONTINUE_WORDS):
        return "continue"
    if any(w in text for w in STOP_WORDS):
        return "end"

    # Numeric fallback — user might say "واحد", "اتنين", "تلاتة"
    if any(w in text for w in ["1", "واحد"]):
        return "break"
    if any(w in text for w in ["2", "اتنين", "اثنين"]):
        return "continue"
    if any(w in text for w in ["3", "تلاتة", "ثلاثة"]):
        return "end"

    return None