#!/bin/bash
docker rm -f av-service
docker build -t slashfury/audio-vtuber-service .
docker run --rm -it --name av-service -p 7860:7860 slashfury/audio-vtuber-service
