import os
import queue
import threading
import tkinter as tk
import pyaudio
import json
import urllib.request
import zipfile
from vosk import Model, KaldiRecognizer
from googletrans import Translator 

# ------------------- CONFIG -------------------
RATE = 16000
CHUNK = 8000  # 0.5 sec chunks
#MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
#MODEL_ZIP = "vosk-model-small-en-us-0.15.zip"
MODEL_DIR = "vosk-model-small-en-us-0.15"
audio_queue = queue.Queue()
translator = Translator()
# --------------------------------------------


def download_and_extract_model():
    """Download and extract the Vosk model if not already present."""
    if not os.path.exists(MODEL_DIR):
        print(" Downloading Vosk model (50MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_ZIP)
        print(" Download complete, extracting...")

        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(".")

        print(" Model extracted.")
    else:
        print(" Vosk model already present.")


class SpeechApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Speech Transcription - Vosk + Translation")
        self.root.geometry("700x500")
        self.root.configure(bg="white")

        self.live_label = tk.Label(
            root, text="Listening...", font=("Arial", 14), bg="white", fg="blue"
        )
        self.live_label.pack(pady=10)

        self.text_box = tk.Text(
            root, wrap="word", font=("Arial", 12), height=20, width=80
        )
        self.text_box.pack(padx=10, pady=10)

        self.stop_button = tk.Button(
            root, text="Stop", command=self.stop, bg="red", fg="white"
        )
        self.stop_button.pack(pady=5)

        self.running = True

    def update_live(self, text):
        self.live_label.config(text="Live: " + text)

    def add_final(self, english_text, hindi_text=None):
        self.text_box.insert(tk.END, "English: " + english_text + "\n")
        if hindi_text:
            self.text_box.insert(tk.END, "Hindi: " + hindi_text + "\n")
        self.text_box.insert(tk.END, "-" * 50 + "\n")
        self.text_box.see(tk.END)

    def stop(self):
        self.running = False
        audio_queue.put(None)
        self.root.quit()


def mic_callback(in_data, frame_count, time_info, status_flags):
    """Called by PyAudio for each audio chunk."""
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)


def recognize(app):
    mic = pyaudio.PyAudio()
    stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=mic_callback,
    )
    stream.start_stream()
    app.add_final("🎤 Speak now...", "🎤 अब बोलें...")

    model = Model(MODEL_DIR)
    recognizer = KaldiRecognizer(model, RATE)
    recognizer.SetWords(True)

    while app.running:
        try:
            data = audio_queue.get(timeout=1)
            if data is None:
                break

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                if "text" in result and result["text"]:
                    english_text = result["text"]
                    try:
                        hindi_text = translator.translate(english_text, src="en", dest="hi").text
                    except Exception:
                        hindi_text = "(Translation failed)"
                    app.add_final(english_text, hindi_text)
            else:
                partial = json.loads(recognizer.PartialResult())
                if "partial" in partial and partial["partial"]:
                    app.update_live(partial["partial"])

        except queue.Empty:
            continue

    stream.stop_stream()
    stream.close()
    mic.terminate()


def main():
    # Ensure model is available
    download_and_extract_model()

    # Start GUI
    root = tk.Tk()
    app = SpeechApp(root)
    threading.Thread(target=recognize, args=(app,), daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()