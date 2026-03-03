# 🎙️ Shulker AI - Speech Recognition & Summarizer API
A production-ready Flask backend that transcribes meeting audio, translates speech to Hindi, and generates AI-powered meeting summaries using Vosk and Google Gemini. The core AI engine of the Shulker meeting ecosystem.

💡 **Made by [Vasu Goel](https://github.com/vasug27)**

---

## ✅ Overview

A production-ready **Python + Flask backend** for real-time speech processing and summarization:
- Transcribes audio files to English text using **Vosk** (offline, no API cost)
- Translates transcription to **Hindi** in real-time via Google Translate
- Generates structured **meeting summaries** with key points and action items using **Gemini**
- Combines transcription + summarization in a single endpoint
- Converts any audio format to WAV via **FFmpeg** before processing
- Deployed via **Docker** on **Render**

---

## 🛠 Tech Stack

| Category | Technologies Used |
|---|---|
| Backend | Python, Flask 3.0.3 |
| Speech Recognition | Vosk 0.3.45 (`vosk-model-small-en-us-0.15`) |
| Audio Processing | FFmpeg, Wave, KaldiRecognizer |
| Translation | googletrans 4.0.0-rc1 |
| AI Summarization | Google Gemini (`gemini-flash-latest`) |
| CORS | Flask-Cors |
| Environment | python-dotenv |
| Containerization | Docker (python:3.12.4-slim) |
| Deployment | Render (Docker web service) |
| Production Server | Gunicorn 23.0.0 |

---

## 📁 Folder Structure

```
Shulker_AI/
├── api.py                          # Flask app - routes, Vosk recognition, Gemini summarization
├── requirements.txt                # Python dependencies
├── dockerfile                      # Docker build config
├── render.yaml                     # Render deployment config
├── Procfile                        # Gunicorn process definition
├── runtime.txt                     # Python 3.12.4
├── .env                            # API key (not committed)
├── .gitignore                      # Ignores venv, .env, __pycache__
└── vosk-model-small-en-us-0.15/    # Offline Vosk English model
    ├── am/                         # Acoustic model
    ├── graph/                      # Language graph (FST)
    ├── ivector/                    # Speaker adaptation vectors
    └── conf/                       # MFCC and model config
```

---

## ⚙️ Setup Guide

**1. Clone the repository**
```bash
git clone https://github.com/vasug27/Shulker_AI.git
cd Shulker_AI
```

**2. Create and activate a virtual environment**
```bash
python -m venv myenv

# macOS/Linux
source myenv/bin/activate

# Windows
myenv\Scripts\activate
```

**3. Install FFmpeg**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
Download from https://ffmpeg.org/download.html
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Configure environment**

Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```
Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

**6. Start the server**
```bash
# Development
python api.py

# Production
gunicorn app:app
```
Server runs at `http://localhost:5000`

**Or run with Docker**
```bash
# Build the image
docker build -t shulker-ai .

# Run the container
docker run -p 5000:5000 --env-file .env shulker-ai
```
Server runs at `http://localhost:5000`

🌐 **Live:** https://shulker-ai.onrender.com

---

## 🎙️ Vosk Model

This repo includes `vosk-model-small-en-us-0.15` - a lightweight offline English speech recognition model.

| Property | Detail |
|---|---|
| Size | Small (mobile-optimized) |
| Sample Rate | 16000 Hz |
| Word Error Rate | 10.38% (TED-LIUM) / 9.85% (LibriSpeech) |
| Speed | 0.11x real-time (desktop) |
| Latency | ~0.15s right context |

> No internet required for transcription - Vosk runs fully offline.

---

## 📌 API Routes

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check - lists available routes |
| POST | `/recognize` | Transcribe audio file to English + Hindi |
| POST | `/summarize` | Generate meeting summary from plain text |
| POST | `/recognize-and-summarize` | Transcribe audio and summarize in one call |

**POST `/recognize`**
- **Content-Type:** `multipart/form-data`
- **Body:** audio file (any format - converted to WAV via FFmpeg)
- **Response:**
```json
{
  "partials": ["partial transcript chunks"],
  "final": {
    "english": "full transcribed text",
    "hindi": "हिंदी अनुवाद"
  }
}
```
- **Errors:** `400` no file uploaded · `400` audio conversion failed

**POST `/summarize`**
- **Content-Type:** `text/plain`
- **Body:** Raw meeting transcript text
- **Response:**
```json
{
  "summary": "Short paragraph + numbered action items",
  "input_length": 1024
}
```
- **Errors:** `400` empty body · `500` Gemini generation failed

**POST `/recognize-and-summarize`**
- **Content-Type:** `multipart/form-data`
- **Body:** audio file
- **Response:**
```json
{
  "recognized_text": "full transcribed text",
  "summary": "Short paragraph + numbered action items"
}
```
- **Errors:** `400` no file uploaded · `400` audio conversion failed

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`feature/new-feature`)
3. Commit changes & push
4. Open a PR 🎉

---

## 🧑 Author

**Vasu Goel**

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:vasugoel2754@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vasugoel503/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vasug27)

---

*Built for Shulker (AI Video Conferencing Assistant) - a quiz generation microservice extending this repo lives at [Shulker_RAG](https://github.com/Shulker-000/Shulker_RAG).*
