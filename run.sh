#!/bin/bash
docker rm -f av-service
docker build -t aservice/audio-vtuber-service .
docker run --rm -it --name av-service --env PORT=7777 -p 7777:7777 aservice/audio-vtuber-service
