"""
stt_handler.py
──────────────
Azure Cognitive Services Speech-to-Text handler for Viora focus sessions.

Uses continuous recognition with the ar-EG (Egyptian Arabic) locale.
Designed to run only during an active focus session — started and stopped
by FocusSession, not by the app.

The get_latest_text() method is passed as stt_fn to the attention monitors.
It returns the latest recognized text and clears it, so each result is
consumed exactly once.
"""

import os
import logging
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class STTHandler:

    def __init__(self):
        self._latest_text = None

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

        # Fires every time Azure finalizes a speech segment
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.canceled.connect(self._on_canceled)

        logger.info("STTHandler initialized. Locale: %s | Region: %s", locale, region)

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Start continuous listening. Called by FocusSession.start()."""
        self._recognizer.start_continuous_recognition()
        logger.info("STT started — listening.")

    def stop(self):
        """Stop listening. Called by FocusSession._end_session()."""
        self._recognizer.stop_continuous_recognition()
        logger.info("STT stopped.")

    def get_latest_text(self):
        """
        Returns the latest recognized text and clears it.
        Returns None if nothing was recognized since last call.
        Passed as stt_fn to the attention monitors.
        """
        text           = self._latest_text
        self._latest_text = None
        return text

    # ── Internal callbacks ─────────────────────────────────────────────────────

    def _on_recognized(self, evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text.strip()
            if text:
                logger.debug("Recognized: '%s'", text)
                self._latest_text = text

    def _on_canceled(self, evt):
        logger.warning("STT recognition canceled: %s", evt.reason)
