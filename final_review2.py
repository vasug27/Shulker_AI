import os
import sys
import queue
import threading
import pyaudio
from google.cloud import speech
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from googletrans import Translator

sys.stdout.reconfigure(encoding="utf-8")

RATE = 16000
CHUNK = int(RATE / 25)  # Frames per chunk
audio_queue = queue.Queue()

def mic_callback(in_data, frame_count, time_info, status_flags):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def request_generator():
    while True:
        data = audio_queue.get()
        if data is None:
            return
        yield speech.StreamingRecognizeRequest(audio_content=data)

class TranscriptionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Live Transcription (English + Hindi)")
        self.text_area = ScrolledText(self.root, wrap=tk.WORD, width=80, height=30, font=("Arial", 14))
        self.text_area.pack(padx=10, pady=10)
        self.text_area.insert(tk.END, "Speak now...\n\n")
        self.text_area.configure(state='disabled')  

    def update_text(self, english, hindi):
        self.text_area.configure(state='normal')
        self.text_area.insert(tk.END, f"English: {english}\nHindi: {hindi}\n\n")
        self.text_area.see(tk.END)
        self.text_area.configure(state='disabled')

    def run(self):
        self.root.mainloop()

def recognize_speech(gui):
    cred_path = r"C:\Users\Vasu Goel\OneDrive\Desktop\Cognimeet_ML\key.json"
    if not os.path.exists(cred_path):
        print("Credential file not found at:", cred_path)
        return
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

    client = speech.SpeechClient()
    translator = Translator()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US"
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    mic = pyaudio.PyAudio()
    stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=mic_callback
    )
    stream.start_stream()

    try:
        responses = client.streaming_recognize(streaming_config, request_generator())
        for response in responses:
            for result in response.results:
                if result.is_final:
                    transcript_en = result.alternatives[0].transcript
                    transcript_hi = translator.translate(transcript_en, dest='hi').text
                    gui.update_text(transcript_en, transcript_hi)
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        mic.terminate()
        print("Stopped by user.")

if __name__ == "__main__":
    gui = TranscriptionGUI()
    threading.Thread(target=recognize_speech, args=(gui,), daemon=True).start()
    gui.run()