#!/bin/bash
docker rm -f av-service
docker build -t slashfury/audio-vtuber-service .
docker run --rm -it --name av-service --env PORT=7777 -p 7777:7777 slashfury/audio-vtuber-service
