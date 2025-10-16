import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)

app = Flask(__name__)

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
    return jsonify({"message": "Gemini Meeting Summarizer API is running!"})


@app.route("/summarize", methods=["POST"])
def summarize_meeting():
    try:
        text = request.data.decode("utf-8").strip()
        if not text:
            return jsonify({"error": "Empty request body"}), 400

        summary = generate_summary(text)

        return jsonify({
            "summary": summary,
            "input_length": len(text)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)