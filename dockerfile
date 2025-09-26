FROM python:3.12.4-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=5000
EXPOSE 5000
CMD gunicorn --worker-class eventlet -w 1 api:app --bind 0.0.0.0:$PORT --timeout 120