FROM python:3.8.14-slim-bullseye

RUN apt update && \
    apt install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 7860
CMD ["sh", "-c", "python app.py --listen-all --port ${PORT} --head-image commish_mouthy_small.png"]
