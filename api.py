import os
import json
import tempfile
import subprocess
import wave
from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
from googletrans import Translator
from dotenv import load_dotenv
import google.generativeai as genai
from flask_cors import CORS

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
    raise RuntimeError(f"Vosk model not found. Place it in '{MODEL_DIR}' before deploying.")

model = Model(MODEL_DIR)

def convert_to_wav(input_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as f:
        f.write(input_bytes)
        input_path = f.name

    output_path = input_path + ".wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-af", "highpass=f=200, lowpass=f=3000, afftdn",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(input_path)
    return output_path


def generate_summary(text: str) -> str:
    model = genai.GenerativeModel("gemini-flash-latest")
    prompt = (
        "You are a helpful meeting assistant. Summarize the following meeting transcript "
        "into a detailed and comprehensive summary in simple language. "
        "Include all important points, decisions, and action items. "
        "Do not use headings or subheadings. "
        "Number each point clearly (1., 2., 3., etc.).\n\n"
        + text
    )
    response = model.generate_content(prompt)
    return response.text.strip()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Unified Speech + Summarizer API running!",
        "routes": ["/recognize", "/summarize", "/recognize-and-summarize"]
    })


@app.route("/recognize", methods=["POST"])
def recognize_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    audio_bytes = file.read()
    wav_path = convert_to_wav(audio_bytes)

    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != RATE:
        return jsonify({"error": "Audio must be WAV mono PCM16 16k"}), 400

    recognizer = KaldiRecognizer(model, wf.getframerate())
    recognizer.SetWords(True)
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
    os.remove(wav_path)

    return jsonify({
        "partials": partials,
        "final": {
            "english": english_text,
            "hindi": hindi_text
        }
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
    wav_path = convert_to_wav(audio_bytes)

    wf = wave.open(wav_path, "rb")
    recognizer = KaldiRecognizer(model, wf.getframerate())

    text_parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            res = json.loads(recognizer.Result())
            if res.get("text"):
                text_parts.append(res["text"])
    final_res = json.loads(recognizer.FinalResult())
    english_text = final_res.get("text", "") or " ".join(text_parts)

    summary = generate_summary(english_text) if english_text else ""

    wf.close()
    os.remove(wav_path)

    return jsonify({
        "recognized_text": english_text,
        "summary": summary
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)