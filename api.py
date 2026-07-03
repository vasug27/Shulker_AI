import os
import io
import json
import subprocess
import wave
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer
from googletrans import Translator
from dotenv import load_dotenv
import google.generativeai as genai
import psutil
import uvicorn

load_dotenv()
RATE = 16000
MODEL_DIR = "vosk-model-small-en-us-0.15"
CHUNK_FRAMES = 1600  
translator = Translator()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)

app = FastAPI(title="Optimized Speech + Summarizer API")

# Configure CORS to allow all origins as requested
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"Vosk model not found in '{MODEL_DIR}'.")

model = Model(MODEL_DIR)
recognizer = KaldiRecognizer(model, RATE)
recognizer.SetWords(True)

model_gem = genai.GenerativeModel("gemini-flash-latest")

def convert_to_wav_bytes(input_bytes: bytes) -> io.BytesIO:
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-nostdin", "-nostats",
        "-threads", "1",
        "-fflags", "+bitexact",
        "-i", "pipe:0",
        "-ar", str(RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "pipe:1"
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
        "You are a helpful meeting assistant. Summarize the following meeting transcript "
        "in a simple language. First, write a short summary paragraph, capturing the overall meeting. Then provide a list of important "
        "points, decisions and action items, numbered clearly like 1., 2., 3., etc."
        "Do not use headings or subheadings.\n\n"
        + text
    )

    response = model_gem.generate_content(prompt)
    return response.text.strip()


@app.get("/")
async def home():
    return {
        "message": "Optimized Speech + Summarizer API running!",
        "routes": ["/recognize", "/summarize", "/recognize-and-summarize"]
    }


@app.post("/recognize")
async def recognize_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    if len(audio_bytes) < 1000:
        return {"partials": [], "final": {"english": "", "hindi": ""}}

    try:
        wav_buffer = convert_to_wav_bytes(audio_bytes)
        wf = wave.open(wav_buffer, "rb")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Audio conversion failed: {e}"})

    recognizer.Reset()

    partials = []
    last_text = None

    while True:
        data = wf.readframes(CHUNK_FRAMES)

        if not data:
            break

        if len(data) < CHUNK_FRAMES * 2:
            continue

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
    english = final_res.get("text", "").strip() or last_text or ""

    try:
        hindi = translator.translate(english, src="en", dest="hi").text if english else ""
    except Exception as e:
        hindi = f"(Translation failed: {e})"

    wf.close()

    return {
        "partials": partials,
        "final": {"english": english, "hindi": hindi}
    }


@app.post(
    "/summarize",
    openapi_extra={
        "requestBody": {
            "content": {
                "text/plain": {
                    "schema": {
                        "type": "string",
                        "example": "This is a sample meeting transcript to summarize."
                    }
                }
            },
            "required": True
        }
    }
)
async def summarize_text(request: Request):
    body = await request.body()
    text = body.decode().strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Empty request body"})

    try:
        summary = generate_summary(text)
        return {"summary": summary, "input_length": len(text)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/recognize-and-summarize")
async def recognize_and_summarize(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    if len(audio_bytes) < 1000:
        return {"recognized_text": "", "summary": ""}

    try:
        wav_buffer = convert_to_wav_bytes(audio_bytes)
        wf = wave.open(wav_buffer, "rb")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Audio conversion failed: {e}"})

    recognizer.Reset()

    text_parts = []
    while True:
        data = wf.readframes(CHUNK_FRAMES)

        if not data:
            break
        if len(data) < CHUNK_FRAMES * 2:
            continue

        if recognizer.AcceptWaveform(data):
            res = json.loads(recognizer.Result())
            if res.get("text"):
                text_parts.append(res["text"])

    final_res = json.loads(recognizer.FinalResult())
    english = final_res.get("text", "") or " ".join(text_parts)
    wf.close()

    summary = ""
    if english:
        try:
            summary = generate_summary(english)
        except Exception:
            summary = "(Summary generation failed)"

    return {"recognized_text": english, "summary": summary}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=False)