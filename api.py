import os
import json
from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
from googletrans import Translator

RATE = 16000
MODEL_DIR = "vosk-model-small-en-us-0.15"
translator = Translator()

app = Flask(__name__)

if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"Vosk model not found. Place it in '{MODEL_DIR}' before deploying.")

model = Model(MODEL_DIR)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Speech-to-Text API with Hindi Translation is running!"})


@app.route("/recognize", methods=["POST"])
def recognize_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    audio_bytes = file.read()

    recognizer = KaldiRecognizer(model, RATE)

    if recognizer.AcceptWaveform(audio_bytes):
        result = json.loads(recognizer.Result())
        text = result.get("text", "")
        hindi = translator.translate(text, src="en", dest="hi").text if text else ""
        return jsonify({"english": text, "hindi": hindi})
    else:
        return jsonify(json.loads(recognizer.PartialResult()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)