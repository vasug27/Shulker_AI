import os
import io
import json
import subprocess
import wave
from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
from googletrans import Translator
from dotenv import load_dotenv
import google.generativeai as genai
from flask_cors import CORS
import psutil

load_dotenv()
RATE = 16000
MODEL_DIR = "vosk-model-small-en-us-0.15"
translator = Translator()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"Vosk model not found in '{MODEL_DIR}'.")

model = Model(MODEL_DIR)
recognizer = KaldiRecognizer(model, RATE)
model_gem = genai.GenerativeModel("gemini-flash-latest")

def convert_to_wav_bytes(input_bytes: bytes) -> io.BytesIO:
    """Convert uploaded audio to WAV PCM16 mono 16kHz entirely in memory."""
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-nostdin", "-nostats",
        "-threads", "1",
        "-i", "pipe:0",
        "-ar", str(RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-f", "wav", "pipe:1"
    ]

    process = subprocess.run(
        cmd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if process.returncode != 0 or len(process.stdout) == 0:
        raise ValueError(f"FFmpeg failed: {process.stderr.decode()}")

    return io.BytesIO(process.stdout)

def generate_summary(text: str) -> str:
    prompt = (
        "You are a helpful meeting assistant. Summarize the following meeting transcript in simple language. "
        "First, write a short summary paragraph capturing the overall meeting. "
        "Then, provide a detailed list of all important points, decisions, timelines and action items, numbered clearly like 1., 2., 3., etc. "
        "Do not use headings or subheadings.\n\n"
        + text
    )
    response = model_gem.generate_content(prompt)
    return response.text.strip()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Optimized Speech + Summarizer API running!",
        "routes": ["/recognize", "/summarize", "/recognize-and-summarize"]
    })

@app.route("/recognize", methods=["POST"])
def recognize_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    audio_bytes = file.read()
    if len(audio_bytes) < 1000:
        return jsonify({
            "partials": [],
            "final": {"english": "", "hindi": ""}
        }), 200

    try:
        wav_buffer = convert_to_wav_bytes(audio_bytes)
        wf = wave.open(wav_buffer, "rb")
    except Exception as e:
        err_msg = str(e).lower()
        print(f"[WARN] Audio conversion failed: {e}")

        if "empty" in err_msg or "no such file" in err_msg:
            return jsonify({"partials": [], "final": {"english": "", "hindi": ""}}), 200

        return jsonify({"error": f"Audio conversion failed: {e}"}), 400

    recognizer.Reset()

    partials, last_text = [], None
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            res = json.loads(recognizer.Result())
            text = res.get("text", "").strip()
            if text and text != last_text:
                partials.append(text)
                last_text = text
        else:
            res = json.loads(recognizer.PartialResult())
            text = res.get("partial", "").strip()
            if text and text != last_text:
                partials.append(text)
                last_text = text

    final_res = json.loads(recognizer.FinalResult())
    english_text = final_res.get("text", "").strip() or last_text or ""
    hindi_text = translator.translate(english_text, src="en", dest="hi").text if english_text else ""

    wf.close()

    print(f"[DEBUG] RAM usage: {psutil.virtual_memory().percent}%")

    return jsonify({
        "partials": partials,
        "final": {"english": english_text, "hindi": hindi_text}
    })

@app.route("/summarize", methods=["POST"])
def summarize_text():
    text = request.data.decode("utf-8").strip()
    if not text:
        return jsonify({"error": "Empty request body"}), 400
    try:
        summary = generate_summary(text)
        return jsonify({"summary": summary, "input_length": len(text)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recognize-and-summarize", methods=["POST"])
def recognize_and_summarize():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    audio_bytes = file.read()
    if len(audio_bytes) < 1000:
        return jsonify({"recognized_text": "", "summary": ""}), 200

    try:
        wav_buffer = convert_to_wav_bytes(audio_bytes)
        wf = wave.open(wav_buffer, "rb")
    except Exception as e:
        err_msg = str(e).lower()
        print(f"[WARN] Audio conversion failed: {e}")

        if "empty" in err_msg or "no such file" in err_msg:
            return jsonify({"recognized_text": "", "summary": ""}), 200

        return jsonify({"error": f"Audio conversion failed: {e}"}), 400

    recognizer.Reset()
    text_parts = []
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if recognizer.AcceptWaveform(data):
            res = json.loads(recognizer.Result())
            if res.get("text"):
                text_parts.append(res["text"])
    final_res = json.loads(recognizer.FinalResult())
    english_text = final_res.get("text", "") or " ".join(text_parts)

    wf.close()

    summary = ""
    if english_text:
        try:
            summary = generate_summary(english_text)
        except Exception:
            summary = "(Summary generation failed)"

    return jsonify({"recognized_text": english_text, "summary": summary})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)