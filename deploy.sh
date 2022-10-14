#!/bin/bash

# Retag docker image ()
docker tag slashfury/audio-vtuber-service gcr.io/audio-vtuber/audio-vtuber-service

# Push image to GCP Artifact Registry. Requires gcloud installed in your environment
# If you don't have gcloud, run `configure_gcp.sh` first
docker push gcr.io/audio-vtuber/audio-vtuber-service
gcloud run deploy audio-vtuber-service --image gcr.io/audio-vtuber/audio-vtuber-service:latest

# Tweak machine allocation with the following command(s)
# gcloud run services update audio-vtuber-service --memory 4Gi --cpu 2
# gcloud run services update audio-vtuber-service --concurrency 4 --max-instances 10
