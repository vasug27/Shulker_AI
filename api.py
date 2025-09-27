import os
import json
import tempfile
import subprocess
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


def convert_to_wav(input_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as input_file:
        input_file.write(input_bytes)
        input_path = input_file.name

    output_path = input_path + ".wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le", 
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    os.remove(input_path)
    return output_path


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Speech-to-Text API with Hindi Translation is running!"})


@app.route("/recognize", methods=["POST"])
def recognize_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    audio_bytes = file.read()

    wav_path = convert_to_wav(audio_bytes)

    recognizer = KaldiRecognizer(model, RATE)
    recognizer.SetWords(True)

    partials = []

    with open(wav_path, "rb") as wf:
        while True:
            data = wf.read(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                if "text" in res and res["text"]:
                    partials.append(res["text"])
            else:
                res = json.loads(recognizer.PartialResult())
                if "partial" in res and res["partial"]:
                    partials.append(res["partial"])

    os.remove(wav_path)

    final_res = json.loads(recognizer.FinalResult())
    english_text = final_res.get("text", "")
    hindi_text = translator.translate(english_text, src="en", dest="hi").text if english_text else ""

    return jsonify({
        "partials": partials,
        "final": {
            "english": english_text,
            "hindi": hindi_text
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
