# EmuSync – Multimodal Emotion Recognition and Prompt Generation System

## 🔍 Overview
EmuSync is a multimodal emotion analysis pipeline designed to detect and synchronize emotional states from both video and audio sources. By aligning per-second emotional data from facial expressions and voice signals, the system generates meaningful emotional summaries and transforms them into Suno-style English prompts using a local LLM (e.g., Deepseek via Ollama).

---

## 🧠 Core Technologies

| Component         | Technology Used                                                 |
|------------------|------------------------------------------------------------------|
| Video Emotion    | OpenCV, DeepFace (MTCNN backend)                                |
| Audio Emotion    | Torchaudio, Librosa, Hugging Face `superb/wav2vec2-large-superb-er` |
| Fusion Logic     | pandas (timestamp alignment and weighted merging)               |
| Prompt Generator | Ollama + Deepseek LLM                                            |
| Output Format    | CSV (intermediate), TXT (final prompts)                         |

Recommended operating system for deployment: **Ubuntu Server** (20.04+).

---

## 📊 Pipeline Steps

1. **Video Emotion Analysis**
   - Process every 10 frames.
   - Aggregate per-second dominant emotions.
   - Store to `video_emotion/emotion_analysis_<timestamp>.csv`.

2. **Audio Emotion Analysis**
   - Split audio into 1-second chunks.
   - Use FSER model to analyze top-2 emotional probabilities.
   - Store to `audio_text_emotion/session_<timestamp>.csv`.

3. **Multimodal Fusion**
   - Align data using second-wise timestamps.
   - Apply weight: video (70%), audio (30%).
   - Merge scores and sum matching emotions.
   - Output top-2 emotions per second.
   - Store to `merge_emotions/merged_emotions.csv`.

4. **Prompt Generation**
   - Send merged emotion data as prompt to Deepseek via Ollama.
   - Generate Suno-compatible English prompt with similar emotional trajectory.
   - Save output as `.txt` under `prompt/` folder with same name as CSV.

---

## 📌 Requirements

- Python 3.8+
- Libraries:
  - `torch`, `transformers`, `pandas`, `opencv-python`, `librosa`, `torchaudio`, `deepface`
- Ollama installed and running with Deepseek model (`ollama run deepseek`)

---

## 📆 Usage

1. Place your input video at `video_input/<filename>.mp4`.
2. Place the separated audio file at `separated_audio/<filename>_audio.mp3`.
3. Run the scripts in order:
   ```bash
   python video_emotion.py
   python audio_text_emotion.py
   python merge_emotions.py
   python generate_prompt.py
   ```
4. Final Suno prompt appears in `prompt/<filename>.txt`

---

## 💰 Output Example
```
neutral 0.67 angry 0.33
happy 0.70 sad 0.30
...
```
Prompt (via LLM):
> A mellow sunrise evolves into an energetic burst of joy, shadowed briefly by moments of doubt. Music reflecting emotional rise, warmth, and subtle melancholy.

---

## 🔧 Author Instructions (DevOps Style)
- Each script is modular, easy to debug independently.
- All CSVs are saved with timestamped filenames.
- Ensure all audio is 16kHz mono WAV (FFmpeg pre-conversion available).
- You may test changes manually or rely on merged CSV preview.
- No CI required; monitor CLI logs for process tracking.

---

## 🚀 Ready to Deploy?
This system is ideal for:
- Emotion-aware audio/video synthesis
- Suno music prompt pipelines
- Real-time emotion summarization

---

📄 Licensed for internal research and demo purposes.

