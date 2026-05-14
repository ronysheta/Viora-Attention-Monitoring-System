import logging
logging.basicConfig(level=logging.INFO)

from session_setup import SessionSetup
from focus_session import FocusSession

def speak(text):
    print(f"[TTS] {text}")

def start_session(settings):
    print(f"[App] Starting session with: {settings}")
    session = FocusSession(
        mode                  = settings["mode"],
        focus_minutes         = settings["focus_mins"],
        break_minutes         = settings["break_mins"],
        accessibility_profile = settings["accessibility_profile"],
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

setup = SessionSetup(
    on_ready  = start_session,
    speak_fn  = speak,
)
setup.begin()

while setup.is_waiting:
    user_input = input(">> ").strip()
    setup.fill_slot(user_input)