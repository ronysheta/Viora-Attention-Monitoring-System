import time
import threading
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from typing import Callable, Optional


# Thresholds

HEAD_YAW_THRESHOLD   = 25   # degrees left/right before counting as distracted
HEAD_PITCH_THRESHOLD = 20   # degrees up/down
EAR_CLOSED_THRESHOLD = 0.20  # below this = eye closed
EAR_CALIBRATION_MIN  = 0.18  # if average EAR during calibration < this,
                             # eyes are closed/covered → disable eye signal

DISTRACTION_SCORE_THRESHOLD = 0.6   # score must exceed this...
DISTRACTION_WINDOW_SECS     = 4     # ...sustained for this many seconds
CALIBRATION_FRAMES          = 30    # frames used to establish eye baseline
NO_FACE_TIMEOUT_SECS        = 10    # seconds without face before fallback alert
RESUME_FACE_FRAMES          = 10    # consecutive frames with face before auto-resume
COOLDOWN_SECS               = 20    # min seconds between distraction triggers

# MediaPipe landmark indices 
# Eye landmarks for EAR calculation (left eye, right eye)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Face points for head pose estimation (nose, chin, eyes, mouth corners)
POSE_POINTS = [1, 152, 33, 263, 61, 291]

# 3D model points matching the above landmarks (generic face model)
MODEL_3D = np.array([
    [0.0,    0.0,    0.0   ],   # nose tip
    [0.0,   -63.6, -12.5  ],   # chin
    [-43.3,  32.7, -26.0  ],   # left eye corner
    [43.3,   32.7, -26.0  ],   # right eye corner
    [-28.9, -28.9, -24.1  ],   # left mouth corner
    [28.9,  -28.9, -24.1  ],   # right mouth corner
], dtype=np.float64)


class CameraAttentionMonitor:
    """
    Camera-based attention monitor using MediaPipe FaceMesh.

    Detects distraction via:
      - Head pose (yaw/pitch) — primary signal, works for all users
      - Eye aspect ratio (EAR) — secondary, auto-enabled only if eyes
        are detected as open during calibration

    Falls back to inactivity timer when no face is detected.

    Drop-in replacement for AttentionMonitor — identical public interface.

    Usage:
        monitor = CameraAttentionMonitor(
            on_distraction=lambda: session.pause(),
            speak_fn=tts.speak,
        )
        monitor.start()
        monitor.register_interaction()
        monitor.stop()
    """

    def __init__(
        self,
        on_distraction: Optional[Callable]       = None,
        on_resume: Optional[Callable]            = None,
        speak_fn: Optional[Callable[[str], None]] = None,
        inactivity_threshold: int                 = 90,
        response_window: int                      = 15,
        checkin_message: str  = "Are you still there? Press any key to continue.",
        alert_message: str    = "It seems you drifted off. Let us refocus.",
        camera_source                             = 0,
        frame_source: Optional[Callable]          = None,
    ):
        self.on_distraction       = on_distraction
        self.on_resume            = on_resume
        self.inactivity_threshold = inactivity_threshold
        self.response_window      = response_window
        self.checkin_message      = checkin_message
        self.alert_message        = alert_message

        self._speak = speak_fn or (lambda t: print(f"[CameraMonitor] TTS: {t}"))

        # Camera
        self._camera_source  = camera_source
        self._frame_source   = frame_source   # optional callable → numpy frame
        self._cap            = None
        self._latest_result = None
        self._result_lock   = threading.Lock()
 
        # MediaPipe
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # State
        self._running              = False
        self._paused               = False 
        self._use_eye_signal       = False  
        self._calibration_done     = False
        self._calibration_ears     = []
        self._last_interaction     = time.time()
        self._last_face_time       = time.time()
        self._waiting_for_response = False
        self._response_received    = False
        self._last_distraction_time = 0      # for cooldown
        self._face_return_frames   = 0       # consecutive frames with face while paused

        # Rolling window of distraction scores
        self._score_window = deque(maxlen=int(
            DISTRACTION_WINDOW_SECS * 10   # ~10 fps expected
        ))

        self._lock          = threading.Lock()
        self._camera_thread = None
        self._inactivity_thread = None

    # Public API

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

        print("[CameraMonitor] Started.")

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        print("[CameraMonitor] Stopped.")

    def register_interaction(self):
        with self._lock:
            self._last_interaction  = time.time()
            self._face_return_frames = 0
 
        if self._paused:
            self._resume_session(triggered_by="keypress")


    # Camera loop

    def _camera_loop(self):
        # Open own camera only if no external frame source provided
        if not self._frame_source:
            source = self._camera_source
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                print(f"[CameraMonitor] Could not open camera: {source}")
                print("[CameraMonitor] Falling back to inactivity-only mode.")
                return

        while self._running:
            frame = self._get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            self._analyze_frame(frame)
            time.sleep(0.1)
            
    def _get_frame(self):
        """Get next frame from own capture or external source."""
        if self._frame_source:
            return self._frame_source()

        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            return frame if ret else None

        return None

    # Frame analysis

    def _analyze_frame(self, frame: np.ndarray):
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            with self._lock:
                self._last_face_time = None
                self._face_return_frames = 0 
            return
        with self._lock:
            self._last_face_time = time.time()
        self._last_face_time = time.time()
        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _   = frame.shape

        # If paused 
        if self._paused:
            with self._lock:
                self._face_return_frames += 1
                frames = self._face_return_frames
 
            if frames >= RESUME_FACE_FRAMES:
                self._resume_session(triggered_by="face_detected")
            return
        
        # Calibration 
        if not self._calibration_done:
            ear = self._compute_ear(landmarks, w, h)
            self._calibration_ears.append(ear)

            if len(self._calibration_ears) >= CALIBRATION_FRAMES:
                avg_ear = np.mean(self._calibration_ears)
                self._use_eye_signal = avg_ear >= EAR_CALIBRATION_MIN
                self._calibration_done = True
                mode = "enabled" if self._use_eye_signal else "disabled (eyes closed/covered)"
                print(f"[CameraMonitor] Calibration done. Eye signal: {mode}. Avg EAR={avg_ear:.3f}")
            return 
        
        # Cooldown check
        with self._lock:
            in_cooldown = (time.time() - self._last_distraction_time) < COOLDOWN_SECS

        if in_cooldown:
            return

        # Score this frame 
        score = self._score_frame(landmarks, w, h)
        self._score_window.append(score)

        if len(self._score_window) == self._score_window.maxlen:
            avg_score = np.mean(self._score_window)
            if avg_score >= DISTRACTION_SCORE_THRESHOLD:
                self._score_window.clear()
                self._trigger_distraction()
    
    # Distraction + Resume
    def _trigger_distraction(self):
        with self._lock:
            if self._paused:
                return
            self._paused                = True
            self._face_return_frames    = 0
            self._last_distraction_time = time.time()
 
        print("[CameraMonitor] Distraction confirmed. Session paused.")
        self._speak(self.checkin_message)
 
        if self.on_distraction:
            self.on_distraction()
 
    def _resume_session(self, triggered_by: str = "unknown"):
        with self._lock:
            if not self._paused:
                return
            self._paused             = False
            self._face_return_frames = 0
            self._last_interaction   = time.time()
            self._score_window.clear()
 
        print(f"[CameraMonitor] Session resumed (triggered by: {triggered_by}).")
        self._speak(self.alert_message)
 
        if self.on_resume:
            self.on_resume()

    # Scoring
    def _score_frame(self, landmarks, w: int, h: int) -> float:
        """
        Combine head pose and (optionally) EAR into a single 0-1 score.
        Higher = more distracted.
        """
        scores = []

        # Head pose — always used
        yaw, pitch = self._compute_head_pose(landmarks, w, h)
        if yaw is not None:
            yaw_score   = min(abs(yaw)   / HEAD_YAW_THRESHOLD,   1.0)
            pitch_score = min(abs(pitch) / HEAD_PITCH_THRESHOLD,  1.0)
            head_score  = max(yaw_score, pitch_score)
            scores.append(("head", head_score, 0.7))   # weight 70%

        # EAR — only if calibration enabled it
        if self._use_eye_signal:
            ear = self._compute_ear(landmarks, w, h)
            ear_score = 1.0 if ear < EAR_CLOSED_THRESHOLD else 0.0
            scores.append(("ear", ear_score, 0.3))       # weight 30%

        if not scores:
            return 0.0

        total_weight = sum(w for _, _, w in scores)
        weighted_sum = sum(s * w for _, s, w in scores)
        return weighted_sum / total_weight

    # Head pose
    def _compute_head_pose(self, landmarks, w: int, h: int):
        """Return (yaw, pitch) in degrees, or (None, None) on failure."""
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
            rmat, _ = cv2.Rodrigues(rvec)
            angles, *_ = cv2.RQDecomp3x3(rmat)

            pitch, yaw = angles[0], angles[1]
            return float(yaw), float(pitch)

        except Exception:
            return None, None

    # Eye aspect ratio 

    def _compute_ear(self, landmarks, w: int, h: int) -> float:
        """Compute average EAR across both eyes."""
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

    # Inactivity fallback 
    # Runs in parallel — catches cases where the camera loses the face entirely (user walks away, dark room, etc.)

    def _inactivity_loop(self):
        while self._running:
            time.sleep(1)

            with self._lock:
                elapsed            = time.time() - self._last_interaction
                paused      = self._paused
                last_face   = self._last_face_time
                in_cooldown = (time.time() - self._last_distraction_time) < COOLDOWN_SECS

            if paused or in_cooldown:
                continue
 
            face_missing = (
                last_face is None or
                time.time() - last_face > NO_FACE_TIMEOUT_SECS
            )
 
            if face_missing and elapsed >= self.inactivity_threshold:
                self._trigger_distraction()


# Quick test 

if __name__ == "__main__":
    def speak(text):
        print(f"[TTS] {text}")
 
    def on_distraction():
        print(">>> SESSION PAUSED — look back at camera or press Enter to resume.\n")
 
    def on_resume():
        print(">>> SESSION RESUMED — back to focusing!\n")
 
    monitor = CameraAttentionMonitor(
        on_distraction=on_distraction,
        on_resume=on_resume,
        speak_fn=speak,
        camera_source=0,
        inactivity_threshold=15,
    )
 
    monitor.start()
    print("Running. Look away to trigger pause. Look back or press Enter to resume. Ctrl+C to stop.\n")
 
    try:
        while True:
            input()
            monitor.register_interaction()
            print("[Test] Interaction registered.")
    except KeyboardInterrupt:
        monitor.stop()
