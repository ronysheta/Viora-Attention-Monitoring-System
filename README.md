# Viora Attention & Focus Monitoring System

A standalone attention monitoring module built for [Viora](https://github.com/ronysheta/Viora-Attention-Monitoring-System) — an AI-powered accessibility assistant for blind and low-vision students.

The module detects study session disengagement through camera-based head tracking and inactivity detection, delivers voice-driven check-ins in Egyptian Arabic, and manages the full session lifecycle including focus blocks, breaks, and session summaries.

---

## Features

- **Two session modes** — Pomodoro (structured blocks + breaks) and Free (continuous session)
- **Camera-based attention detection** — MediaPipe FaceMesh head pose estimation and eye aspect ratio tracking
- **Personal baseline calibration** — adapts to each user's natural sitting position to prevent false alerts
- **Stability suppression** — ignores consistent off-centre angles (e.g. reading a corner monitor)
- **Voice check-in system** — confirms distraction through a two-step voice prompt before pausing the session
- **Inactivity fallback** — works without a camera using silence and interaction detection
- **Three accessibility profiles** — standard, low vision, and blind
- **Egyptian Arabic throughout** — all prompts, check-ins, and session summaries in Egyptian dialect
- **Azure STT integration** — continuous voice recognition with `ar-EG` locale during sessions
- **Voice command routing** — session control via spoken commands (*وقف*, *استراحة*, *كمّل*)
- **User profile persistence** — saves preferences locally and skips setup on return visits
- **Session summary** — saved as JSON and read aloud at the end of every session

---

## Project Structure

```
viora-attention-monitoring/
├── accessibility_profile.py      # Profiles, Egyptian Arabic prompts, user profile save/load
├── attention_monitor.py          # No-camera inactivity monitor
├── camera_attention_monitor.py   # Camera-based head pose and eye tracking monitor
├── focus_session.py              # Session coordinator — states, timer, monitor lifecycle
├── session_setup.py              # Voice-driven session configuration
├── stt_handler.py                # Azure Cognitive Services STT (ar-EG)
├── voice_command_router.py       # Routes STT output to session actions
├── summaries/                    # Auto-generated session JSON summaries
├── user_profile.json             # Saved user preferences (auto-generated on first run)
├── requirements.txt
└── .env                          # API keys (never committed)
```

---

## Requirements

- Python 3.11+
- Webcam (optional — falls back to inactivity detection if unavailable)
- Microphone
- Azure Cognitive Services Speech account

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/ronysheta/Viora-Attention-Monitoring-System.git
cd Viora-Attention-Monitoring-System
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your environment variables**

Create a `.env` file in the project root:
```
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_region
AZURE_SPEECH_LOCALE=ar-EG
TTS_ENDPOINT=your_tts_endpoint
```

---

## Running the Project

```bash
python test_run.py
```

On first run the system will ask:
- Whether you have any accessibility needs
- Which session mode you prefer (Pomodoro or Free)
- How long you want to focus and how long for breaks

These preferences are saved and reused on all future runs.

---

## How It Works

```
Session starts
      ↓
Camera calibrates personal head pose baseline (60 frames)
      ↓
Attention monitor runs continuously in background
      ↓
Distraction suspected (head deviation + low EAR score sustained)
      ↓
Voice check-in played in Egyptian Arabic
"لسه معانا؟ قول أي حاجة نكمل"
      ↓
User responds → session continues
No response   → distraction confirmed → session paused
      ↓
User returns (face detected or voice/key press)
      ↓
Session resumes from exact remaining time
      ↓
Block ends → user chooses break / continue / end via voice
      ↓
Session summary saved as JSON and read aloud
```

---

## Accessibility Profiles

| Profile | Camera | Interaction | Sensitivity |
|---|---|---|---|
| Standard | Enabled | Voice + key presses | Low |
| Low Vision | Enabled | Voice only | Low |
| Blind | Disabled | Voice only | — |

Profiles are detected automatically through a natural voice conversation at first launch and never asked again.

---

## Voice Commands (Egyptian Arabic)

| Command | Action |
|---|---|
| *وقف* / *خلاص* | End session |
| *استراحة* | Take a break (at block end) |
| *كمّل* | Continue to next block (at block end) |
| Any speech | Registers as interaction, resets inactivity timer |

---

## Session Summary Example

```json
{
  "mode": "pomodoro",
  "accessibility_profile": "standard",
  "started_at": "2026-05-15T02:07:00",
  "ended_at": "2026-05-15T02:32:00",
  "blocks_completed": 1,
  "breaks_taken": 1,
  "distraction_count": 1,
  "distraction_timestamps": ["2026-05-15T02:15:32"],
  "total_focus_minutes": 25.0,
  "total_break_minutes": 5.0
}
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture |
| `mediapipe==0.10.11` | Face mesh landmark detection |
| `numpy` | Numerical operations |
| `azure-cognitiveservices-speech` | Azure STT with ar-EG locale |
| `python-dotenv` | Environment variable management |
| `requests` | TTS endpoint communication |

---

## Part of Viora

This module is part of **Viora** — an AI-powered accessibility assistant for blind and low-vision students developed at the Faculty of Computers and Data Science, Alexandria University.

Viora integrates speech interaction, document understanding, OCR, scene understanding, and attention monitoring into a single voice-first Arabic experience.

---

## License

This project is developed as part of a graduation project at Alexandria University. All rights reserved.