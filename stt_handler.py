"""
stt_handler.py
──────────────
Azure Cognitive Services Speech-to-Text handler for Viora focus sessions.

Uses continuous recognition with the ar-EG (Egyptian Arabic) locale.
Designed to run only during an active focus session — started and stopped
by FocusSession, not by the app.

Uses a callback system so multiple consumers (monitor + router) each
receive every STT result independently without consuming each other's text.

Usage:
    stt = STTHandler()

    # Register as many callbacks as needed
    stt.add_callback(monitor.on_speech)
    stt.add_callback(router.on_speech)

    stt.start()
    stt.stop()
"""

import os
import logging
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from typing import Callable, List

load_dotenv()
logger = logging.getLogger(__name__)


class STTHandler:

    def __init__(self):
        self._callbacks: List[Callable[[str], None]] = []

        key    = os.getenv("AZURE_SPEECH_KEY")
        region = os.getenv("AZURE_SPEECH_REGION")
        locale = os.getenv("AZURE_SPEECH_LOCALE", "ar-EG")

        if not key or not region:
            raise ValueError(
                "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env"
            )

        speech_config = speechsdk.SpeechConfig(
            subscription = key,
            region       = region,
        )
        speech_config.speech_recognition_language = locale

        audio_config     = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config = speech_config,
            audio_config  = audio_config,
        )

        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.canceled.connect(self._on_canceled)

        logger.info("STTHandler initialized. Locale: %s | Region: %s", locale, region)

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_callback(self, fn: Callable[[str], None]):
        """
        Register a callback that receives every recognized text result.
        Call this before start() — both monitor and router register here.

        Example:
            stt.add_callback(lambda text: monitor.on_speech(text))
            stt.add_callback(lambda text: router.on_speech(text))
        """
        self._callbacks.append(fn)

    def start(self):
        """Start continuous listening. Called by FocusSession.start()."""
        self._recognizer.start_continuous_recognition()
        logger.info("STT started — listening.")

    def stop(self):
        """Stop listening. Called by FocusSession._end_session()."""
        self._recognizer.stop_continuous_recognition()
        logger.info("STT stopped.")

    # ── Internal callbacks ─────────────────────────────────────────────────────

    def _on_recognized(self, evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text.strip()
            if text:
                logger.debug("Recognized: '%s'", text)
                # Fire all registered callbacks with the same text
                for fn in self._callbacks:
                    try:
                        fn(text)
                    except Exception as e:
                        logger.error("STT callback error: %s", e)

    def _on_canceled(self, evt):
        logger.warning("STT recognition canceled: %s", evt.reason)