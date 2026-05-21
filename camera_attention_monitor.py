"""
camera_attention_monitor.py
───────────────────────────
Camera-based attention monitor for Viora using MediaPipe FaceMesh.

Detects distraction via:
  - Head pose (yaw/pitch) deviation from calibrated personal baseline
  - Eye aspect ratio (EAR) — secondary, auto-enabled only if eyes are
    detected as open during calibration
  - Stability detection — consistent off-centre angle = reading, not distracted
  - STT voice activity — any detected speech resets the inactivity timer
  - Inactivity fallback — when face is lost entirely (walked away, dark room)

Key improvements over v1:
  - Calibrates personal head pose baseline so sitting off-centre or having
    a corner monitor no longer causes false positives
  - Stability suppression — steady angle at a consistent position is ignored
  - STT integration — speech auto-registers as interaction
  - head_pose_weight — can be dialled down for users with limited mobility
  - Voice-driven check-ins — no key press required if key_presses_enabled=False
  - Auto fallback to inactivity-only mode if camera fails to open
  - All prompts sourced from accessibility_profile.PROMPTS in Egyptian Arabic
  - logging instead of print throughout

Drop-in replacement for AttentionMonitor — identical public interface.

Usage:
    monitor = CameraAttentionMonitor(
        on_distraction      = session.on_distraction,
        on_resume           = session.on_resume,
        speak_fn            = tts.speak,
        stt_fn              = stt.get_latest_text,
        head_pose_weight    = 0.7,
        key_presses_enabled = True,
        camera_source       = 0,
    )
    monitor.start()
    monitor.register_interaction()
    monitor.stop()
"""

import time
import logging
import threading
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from typing import Callable, Optional

from accessibility_profile import PROMPTS

logger = logging.getLogger(__name__)


# ── Sensitivity presets ────────────────────────────────────────────────────────
# Set automatically from AccessibilityProfile — never exposed to the user.

SENSITIVITY_PRESETS = {
    "low": {
        "HEAD_YAW_THRESHOLD":          40,
        "HEAD_PITCH_THRESHOLD":        30,
        "DISTRACTION_SCORE_THRESHOLD": 0.70,
        "DISTRACTION_WINDOW_SECS":     8,
        "STABILITY_WINDOW_SECS":       3,
        "STABILITY_STD_THRESHOLD":     6.0,
    },
    "medium": {
        "HEAD_YAW_THRESHOLD":          32,
        "HEAD_PITCH_THRESHOLD":        25,
        "DISTRACTION_SCORE_THRESHOLD": 0.65,
        "DISTRACTION_WINDOW_SECS":     6,
        "STABILITY_WINDOW_SECS":       2,
        "STABILITY_STD_THRESHOLD":     5.0,
    },
    "high": {
        "HEAD_YAW_THRESHOLD":          25,
        "HEAD_PITCH_THRESHOLD":        20,
        "DISTRACTION_SCORE_THRESHOLD": 0.60,
        "DISTRACTION_WINDOW_SECS":     4,
        "STABILITY_WINDOW_SECS":       1,
        "STABILITY_STD_THRESHOLD":     4.0,
    },
}

# ── Fixed constants ────────────────────────────────────────────────────────────
EAR_CLOSED_THRESHOLD = 0.20
EAR_CALIBRATION_MIN  = 0.18
CALIBRATION_FRAMES   = 60
NO_FACE_TIMEOUT_SECS = 10
RESUME_FACE_FRAMES   = 10
COOLDOWN_SECS        = 20

# ── MediaPipe landmark indices ─────────────────────────────────────────────────
LEFT_EYE    = [33, 160, 158, 133, 153, 144]
RIGHT_EYE   = [362, 385, 387, 263, 373, 380]
POSE_POINTS = [1, 152, 33, 263, 61, 291]

MODEL_3D = np.array([
    [ 0.0,    0.0,    0.0  ],
    [ 0.0,  -63.6,  -12.5 ],
    [-43.3,  32.7,  -26.0 ],
    [ 43.3,  32.7,  -26.0 ],
    [-28.9, -28.9,  -24.1 ],
    [ 28.9, -28.9,  -24.1 ],
], dtype=np.float64)


class CameraAttentionMonitor:

    def __init__(
        self,
        on_distraction: Optional[Callable] = None,
        on_resume: Optional[Callable] = None,
        speak_fn: Optional[Callable[[str], None]] = None,
        stt_fn: Optional[Callable[[], Optional[str]]] = None,
        inactivity_threshold: int = 90,
        response_window: int = 15,
        key_presses_enabled: bool = True,
        head_pose_weight: float = 0.7,
        sensitivity: str = "low",
        camera_source = 0,
        frame_source: Optional[Callable] = None,
        _tracking_paused = False,
    ):
        """
        Parameters
        ----------
        on_distraction : callable
            Fired when distraction is confirmed.
        on_resume : callable
            Fired when user returns after a pause.
        speak_fn : callable(str)
            TTS function — called with Egyptian Arabic text.
        stt_fn : callable() -> str | None
            Returns latest STT-detected text or None.
            Any non-None return auto-registers as interaction.
        inactivity_threshold : int
            Seconds without face/interaction before fallback alert.
        response_window : int
            Seconds to wait for check-in response.
        key_presses_enabled : bool
            If False, check-in never mentions pressing anything.
        head_pose_weight : float
            0.0-1.0. Lower values reduce reliance on head pose signal.
            Set automatically from AccessibilityProfile.
        sensitivity : str
            "low" | "medium" | "high" — set from AccessibilityProfile.
        camera_source : int or str
            Camera index or DroidCam URL.
        frame_source : callable() -> np.ndarray | None
            Optional external frame provider (overrides camera_source).
        """
        self.on_distraction       = on_distraction
        self.on_resume            = on_resume
        self.inactivity_threshold = inactivity_threshold
        self.response_window      = response_window
        self.key_presses_enabled  = key_presses_enabled
        self.head_pose_weight     = max(0.0, min(1.0, head_pose_weight))

        self._speak_fn = speak_fn or (lambda t: logger.info("[TTS] %s", t))
        self._stt_fn   = stt_fn

        # ── Sensitivity preset ─────────────────────────────────────────────────
        assert sensitivity in SENSITIVITY_PRESETS, \
            f"sensitivity must be one of {list(SENSITIVITY_PRESETS)}"
        p = SENSITIVITY_PRESETS[sensitivity]
        self._yaw_thresh       = p["HEAD_YAW_THRESHOLD"]
        self._pitch_thresh     = p["HEAD_PITCH_THRESHOLD"]
        self._score_thresh     = p["DISTRACTION_SCORE_THRESHOLD"]
        self._window_secs      = p["DISTRACTION_WINDOW_SECS"]
        self._stability_window = p["STABILITY_WINDOW_SECS"]
        self._stability_std    = p["STABILITY_STD_THRESHOLD"]
        logger.info(
            "Sensitivity: %s | yaw+-%d pitch+-%d window=%ds",
            sensitivity, self._yaw_thresh, self._pitch_thresh, self._window_secs
        )

        # ── Camera ─────────────────────────────────────────────────────────────
        self._camera_source = camera_source
        self._frame_source  = frame_source
        self._cap           = None
        self._camera_ok     = False

        # ── MediaPipe ──────────────────────────────────────────────────────────
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # ── Calibration ────────────────────────────────────────────────────────
        self._calibration_done    = False
        self._calibration_ears    = []
        self._calibration_yaws    = []
        self._calibration_pitches = []
        self._use_eye_signal      = False
        self._baseline_yaw        = 0.0
        self._baseline_pitch      = 0.0

        # ── Angle history for stability detection ──────────────────────────────
        _stability_maxlen   = int(self._stability_window * 10)
        self._yaw_history   = deque(maxlen=_stability_maxlen)
        self._pitch_history = deque(maxlen=_stability_maxlen)

        # ── Distraction scoring window ─────────────────────────────────────────
        self._score_window = deque(maxlen=int(self._window_secs * 10))

        # ── Session state ──────────────────────────────────────────────────────
        self._running               = False
        self._paused                = False
        self._waiting_for_response  = False
        self._response_received     = False
        self._last_interaction      = time.time()
        self._last_face_time        = time.time()
        self._last_distraction_time = 0
        self._face_return_frames    = 0

        self._lock              = threading.Lock()
        self._camera_thread     = None
        self._inactivity_thread = None
        self._stt_thread        = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._running          = True
        self._last_interaction = time.time()
        self._last_face_time   = time.time()

        self._camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True
        )
        self._camera_thread.start()

        self._inactivity_thread = threading.Thread(
            target=self._inactivity_loop, daemon=True
        )
        self._inactivity_thread.start()

        if self._stt_fn:
            self._stt_thread = threading.Thread(
                target=self._stt_loop, daemon=True
            )
            self._stt_thread.start()

        logger.info("CameraAttentionMonitor started.")

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        logger.info("CameraAttentionMonitor stopped.")

    def register_interaction(self):
        """
        Call on any key press or button tap.
        Also called automatically by STT loop when speech is detected.
        Resets inactivity timer. If paused, resumes the session.
        """
        was_paused = False

        with self._lock:
            self._last_interaction   = time.time()
            self._face_return_frames = 0

            if self._waiting_for_response:
                self._response_received    = True
                self._waiting_for_response = False

            if self._paused:
                self._paused = False
                was_paused   = True

        if was_paused:
            self._resume_session(triggered_by="interaction")
    
    def pause_tracking(self):
        """Pause distraction detection during breaks."""
        with self._lock:
            self._tracking_paused = True
            self._score_window.clear()
        logger.info("Attention tracking paused.")

    def resume_tracking(self):
        """Resume distraction detection when focus block starts."""
        with self._lock:
            self._tracking_paused  = False
            self._last_interaction = time.time()
            self._score_window.clear()
        logger.info("Attention tracking resumed.")

    # ── STT polling loop ───────────────────────────────────────────────────────

    def _stt_loop(self):
        """Poll STT every 0.5s. Any detected speech = register interaction."""
        while self._running:
            time.sleep(0.5)
            try:
                text = self._stt_fn()
            except Exception as e:
                logger.error("STT error: %s", e)
                continue
            if text:
                logger.debug("STT detected: '%s'", text)
                self.register_interaction()

    # ── Camera loop ────────────────────────────────────────────────────────────

    def _camera_loop(self):
        if not self._frame_source:
            self._cap = cv2.VideoCapture(self._camera_source)
            if not self._cap.isOpened():
                logger.warning(
                    "Could not open camera: %s. Falling back to inactivity-only mode.",
                    self._camera_source
                )
                self._speak(PROMPTS["camera_failed"])
                self._camera_ok = False
                return

        self._camera_ok = True
        self._speak(PROMPTS["calibrating"])

        while self._running:
            frame = self._get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            self._analyze_frame(frame)
            time.sleep(0.1)

    def _get_frame(self):
        if self._frame_source:
            return self._frame_source()
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            return frame if ret else None
        return None

    # ── Frame analysis ─────────────────────────────────────────────────────────

    def _analyze_frame(self, frame: np.ndarray):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            with self._lock:
                self._last_face_time     = None
                self._face_return_frames = 0
            return

        with self._lock:
            self._last_face_time = time.time()

        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _   = frame.shape

        # Resume check while paused
        if self._paused:
            with self._lock:
                self._face_return_frames += 1
                frames = self._face_return_frames
            if frames >= RESUME_FACE_FRAMES:
                self._resume_session(triggered_by="face_detected")
            return

        # Calibration phase
        if not self._calibration_done:
            self._collect_calibration(landmarks, w, h)
            return

        # Cooldown check
        with self._lock:
            in_cooldown = (time.time() - self._last_distraction_time) < COOLDOWN_SECS
        if in_cooldown:
            return
        # Don't score during breaks
        with self._lock:
            if self._tracking_paused:
                return
        # Score and check window
        score = self._score_frame(landmarks, w, h)
        self._score_window.append(score)

        if len(self._score_window) == self._score_window.maxlen:
            avg_score = np.mean(self._score_window)
            if avg_score >= self._score_thresh:
                self._score_window.clear()
                self._trigger_checkin()

    # ── Calibration ────────────────────────────────────────────────────────────

    def _collect_calibration(self, landmarks, w: int, h: int):
        ear = self._compute_ear(landmarks, w, h)
        self._calibration_ears.append(ear)

        yaw, pitch = self._compute_head_pose(landmarks, w, h)
        if yaw is not None:
            self._calibration_yaws.append(yaw)
            self._calibration_pitches.append(pitch)

        if len(self._calibration_ears) >= CALIBRATION_FRAMES:
            avg_ear = np.mean(self._calibration_ears)
            self._use_eye_signal = avg_ear >= EAR_CALIBRATION_MIN

            if self._calibration_yaws:
                self._baseline_yaw   = float(np.median(self._calibration_yaws))
                self._baseline_pitch = float(np.median(self._calibration_pitches))

            self._calibration_done = True
            logger.info(
                "Calibration done. Eye: %s. EAR=%.3f. Baseline yaw=%.1f pitch=%.1f",
                "on" if self._use_eye_signal else "off",
                avg_ear, self._baseline_yaw, self._baseline_pitch,
            )
            self._speak(PROMPTS["calibration_done"])

    # ── Check-in flow ──────────────────────────────────────────────────────────

    def _trigger_checkin(self):
        """
        Camera suspects distraction — play voice check-in and wait for
        STT or key press response before confirming distraction.
        This prevents false positives from brief glances away.
        """
        with self._lock:
            if self._paused:
                return
            self._waiting_for_response = True
            self._response_received    = False

        logger.info("Distraction suspected. Playing check-in.")
        self._speak(PROMPTS["checkin_camera"])

        deadline = time.time() + self.response_window
        while time.time() < deadline:
            time.sleep(0.5)
            with self._lock:
                if self._response_received:
                    self._last_interaction = time.time()
                    logger.info("User responded to check-in.")
                    return

        # No response — confirm distraction
        with self._lock:
            self._waiting_for_response  = False
            self._paused                = True
            self._face_return_frames    = 0
            self._last_distraction_time = time.time()

        logger.info("No check-in response. Distraction confirmed.")
        self._speak(PROMPTS["distraction_confirmed"])

        if self.on_distraction:
            self.on_distraction()

    # ── Resume ─────────────────────────────────────────────────────────────────

    def _resume_session(self, triggered_by: str = "unknown"):
        with self._lock:
            if not self._paused:
                return
            self._paused             = False
            self._face_return_frames = 0
            self._last_interaction   = time.time()
            self._score_window.clear()

        logger.info("Session resumed (triggered by: %s).", triggered_by)
        self._speak(PROMPTS["resume_prompt"])

        if self.on_resume:
            self.on_resume()

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score_frame(self, landmarks, w: int, h: int) -> float:
        scores = []

        yaw, pitch = self._compute_head_pose(landmarks, w, h)
        if yaw is not None and self.head_pose_weight > 0:
            yaw_dev   = yaw   - self._baseline_yaw
            pitch_dev = pitch - self._baseline_pitch

            yaw_score   = min(abs(yaw_dev)   / self._yaw_thresh,   1.0)
            pitch_score = min(abs(pitch_dev) / self._pitch_thresh,  1.0)
            head_score  = max(yaw_score, pitch_score)

            self._yaw_history.append(yaw_dev)
            self._pitch_history.append(pitch_dev)

            # Stability suppression
            if (len(self._yaw_history) == self._yaw_history.maxlen and
                    len(self._pitch_history) == self._pitch_history.maxlen):
                if (float(np.std(self._yaw_history))   < self._stability_std and
                        float(np.std(self._pitch_history)) < self._stability_std):
                    head_score *= 0.5
                    logger.debug("Stable head angle — score suppressed.")

            scores.append(("head", head_score, self.head_pose_weight))

        if self._use_eye_signal:
            ear        = self._compute_ear(landmarks, w, h)
            ear_score  = 1.0 if ear < EAR_CLOSED_THRESHOLD else 0.0
            ear_weight = 1.0 - self.head_pose_weight
            scores.append(("ear", ear_score, ear_weight))

        if not scores:
            return 0.0

        total_weight = sum(wt for _, _, wt in scores)
        weighted_sum = sum(s * wt for _, s, wt in scores)
        return weighted_sum / total_weight

    # ── Head pose ──────────────────────────────────────────────────────────────

    def _compute_head_pose(self, landmarks, w: int, h: int):
        try:
            image_points = np.array([
                [landmarks[i].x * w, landmarks[i].y * h]
                for i in POSE_POINTS
            ], dtype=np.float64)

            focal   = w
            center  = (w / 2, h / 2)
            cam_mat = np.array([
                [focal, 0,     center[0]],
                [0,     focal, center[1]],
                [0,     0,     1        ],
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            _, rvec, _ = cv2.solvePnP(
                MODEL_3D, image_points, cam_mat, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            rmat, _    = cv2.Rodrigues(rvec)
            angles, *_ = cv2.RQDecomp3x3(rmat)
            pitch, yaw = angles[0], angles[1]
            return float(yaw), float(pitch)

        except Exception:
            return None, None

    # ── Eye aspect ratio ───────────────────────────────────────────────────────

    def _compute_ear(self, landmarks, w: int, h: int) -> float:
        def eye_ear(indices):
            pts = np.array([
                [landmarks[i].x * w, landmarks[i].y * h]
                for i in indices
            ])
            A = np.linalg.norm(pts[1] - pts[5])
            B = np.linalg.norm(pts[2] - pts[4])
            C = np.linalg.norm(pts[0] - pts[3])
            return (A + B) / (2.0 * C) if C > 0 else 0.3

        return (eye_ear(LEFT_EYE) + eye_ear(RIGHT_EYE)) / 2.0

    # ── Inactivity fallback ────────────────────────────────────────────────────

    def _inactivity_loop(self):
        """
        Parallel fallback — triggers when face is lost entirely
        (walked away, dark room, camera turned off mid-session).
        """
        while self._running:
            time.sleep(1)

            with self._lock:
                elapsed     = time.time() - self._last_interaction
                paused      = self._paused
                tracking_paused  = self._tracking_paused
                last_face   = self._last_face_time
                in_cooldown = (time.time() - self._last_distraction_time) < COOLDOWN_SECS

            if paused or in_cooldown or tracking_paused:
                continue

            face_missing = (
                last_face is None or
                time.time() - last_face > NO_FACE_TIMEOUT_SECS
            )

            if face_missing and elapsed >= self.inactivity_threshold:
                logger.info("Face lost and inactivity threshold reached.")
                self._trigger_checkin()

    # ── TTS helper ─────────────────────────────────────────────────────────────

    def _speak(self, text: str):
        try:
            self._speak_fn(text)
        except Exception as e:
            logger.error("TTS error: %s", e)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    def speak(text):
        print(f"[TTS] {text}")

    def on_distraction():
        print(">>> SESSION PAUSED\n")

    def on_resume():
        print(">>> SESSION RESUMED\n")

    monitor = CameraAttentionMonitor(
        on_distraction       = on_distraction,
        on_resume            = on_resume,
        speak_fn             = speak,
        camera_source        = 0,
        inactivity_threshold = 15,
        sensitivity          = "low",
        head_pose_weight     = 0.7,
        key_presses_enabled  = True,
    )

    monitor.start()
    print("Running. Look away to trigger check-in. Ctrl+C to stop.\n")

    try:
        while True:
            input()
            monitor.register_interaction()
            print("[Test] Interaction registered.")
    except KeyboardInterrupt:
        monitor.stop()