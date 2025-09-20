import os
import queue
import threading
from flask import Flask, request, jsonify, Response
import pyaudio
from google.cloud import speech
from google.cloud import translate_v2 as translate

RATE = 16000
CHUNK = int(RATE / 40)
audio_queue = queue.Queue()

cred_path = r"C:\Users\Vasu Goel\OneDrive\Desktop\Cognimeet_ML\key.json"
if not os.path.exists(cred_path):
    raise FileNotFoundError(f"Credential file not found at: {cred_path}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

speech_client = speech.SpeechClient()
translator = translate.Client()

def mic_callback(in_data, frame_count, time_info, status_flags):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def request_generator():
    while True:
        data = audio_queue.get()
        if data is None:
            return
        yield speech.StreamingRecognizeRequest(audio_content=data)

app = Flask(__name__)

@app.route("/start_transcription", methods=["POST"])
def start_transcription():
    """
    Starts live transcription. Expects audio stream from client.
    Returns JSON with live interim English and final English + Hindi.
    """
    responses_list = []

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
        for response in speech_client.streaming_recognize(streaming_config, request_generator()):
            for result in response.results:
                transcript_en = result.alternatives[0].transcript
                if result.is_final:
                    try:
                        transcript_hi = translator.translate(transcript_en, target_language="hi")["translatedText"]
                    except Exception:
                        transcript_hi = transcript_en
                    responses_list.append({
                        "interim": False,
                        "english": transcript_en,
                        "hindi": transcript_hi
                    })
                else:
                    responses_list.append({
                        "interim": True,
                        "english": transcript_en
                    })
            yield f"data: {responses_list[-1]}\n\n"

    except Exception as e:
        yield f"data: {{'error': '{e}'}}\n\n"
    finally:
        stream.stop_stream()
        stream.close()
        mic.terminate()

    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)