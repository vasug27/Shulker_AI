import os
import json
from vosk import Model, KaldiRecognizer
from flask import Flask
from flask_socketio import SocketIO, emit
from googletrans import Translator

RATE = 16000
MODEL_DIR = "vosk-model-small-en-us-0.15"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
translator = Translator()

if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"Vosk model not found in '{MODEL_DIR}'")

model = Model(MODEL_DIR)


@socketio.on("start")
def handle_start(data):
    global recognizer
    recognizer = KaldiRecognizer(model, RATE)
    recognizer.SetWords(True)
    emit("status", {"message": "Recognition started"})


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        text = result.get("text", "")
        if text:
            hindi = translator.translate(text, src="en", dest="hi").text
            emit("final", {"english": text, "hindi": hindi})
    else:
        partial = json.loads(recognizer.PartialResult())
        if "partial" in partial:
            emit("partial", {"english": partial["partial"]})


@socketio.on("stop")
def handle_stop():
    final = json.loads(recognizer.FinalResult())
    text = final.get("text", "")
    hindi = translator.translate(text, src="en", dest="hi").text if text else ""
    emit("final", {"english": text, "hindi": hindi})
    emit("status", {"message": "Recognition stopped"})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
