import os
import sys
import queue
import threading
import tkinter as tk
import pyaudio
from google.cloud import speech
from google.cloud import translate_v2 as translate

sys.stdout.reconfigure(encoding="utf-8")

RATE = 16000                   # Sample rate (Hz)
CHUNK = int(RATE / 40)         # Frames per chunk
audio_queue = queue.Queue()

# Callback: called by PyAudio whenever a new chunk of mic audio is available
def mic_callback(in_data, frame_count, time_info, status_flags):
    audio_queue.put(in_data)            
    return (None, pyaudio.paContinue)     # Continue streaming

# Generator function: yields audio chunks from queue to Google Speech API
def request_generator():
    while True:
        data = audio_queue.get()         
        if data is None:               
            return
        yield speech.StreamingRecognizeRequest(audio_content=data)

class SpeechApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Speech Translation")
        self.root.geometry("600x400")
        self.root.configure(bg="white")

        self.live_label = tk.Label(root, text="Listening...", font=("Arial", 14), bg="white", fg="blue")
        self.live_label.pack(pady=10)

        self.text_box = tk.Text(root, wrap="word", font=("Arial", 12), height=15, width=70)
        self.text_box.pack(padx=10, pady=10)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop, bg="red", fg="white")
        self.stop_button.pack(pady=5)

        self.running = True

    def update_live(self, text):
        self.live_label.config(text="Live: " + text)

    def add_final(self, text):
        self.text_box.insert(tk.END, "Final: " + text + "\n")
        self.text_box.see(tk.END)

    def stop(self):
        self.running = False
        audio_queue.put(None) 
        self.root.quit()

def recognize(app):
    cred_path = r"C:\Users\Vasu Goel\OneDrive\Desktop\Cognimeet_ML\key.json"
    if not os.path.exists(cred_path):
        app.add_final("Credential file not found!")
        return
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

    # Create API clients
    client = speech.SpeechClient()
    translator = translate.Client()

    # Recognition settings
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # 16-bit PCM
        sample_rate_hertz=RATE,
        language_code="en-US" 
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True 
    )

    # ---------- Microphone Stream ----------
    mic = pyaudio.PyAudio()
    stream = mic.open(
        format=pyaudio.paInt16,   
        channels=1,               
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=mic_callback,  # Pushes data to queue
    )
    stream.start_stream()

    app.add_final("🎤 Speak now...")

    try:
        responses = client.streaming_recognize(streaming_config, request_generator())

        for response in responses:
            if not app.running:
                break

            for result in response.results:
                transcript = result.alternatives[0].transcript  # Best guess

                if result.is_final: 
                    try:
                        translated = translator.translate(transcript, target_language="en")["translatedText"]
                    except Exception:
                        translated = transcript
                    app.add_final(translated)
                else:
                    app.update_live(transcript)

    except Exception as e:
        app.add_final(f"Error: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        mic.terminate()

def main():
    root = tk.Tk()
    app = SpeechApp(root)

    threading.Thread(target=recognize, args=(app,), daemon=True).start()

    # Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()