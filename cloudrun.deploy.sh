#!/usr/bin/env bash
set -euo pipefail
REGION=us-central1
REPO=ml-repo
PROJ=$(gcloud config get-value project)
IMAGE="${REGION}-docker.pkg.dev/${PROJ}/${REPO}/image-clf-backend:$(date +%Y%m%d%H%M)"
gcloud builds submit --tag "$IMAGE" .
gcloud run deploy image-clf-backend --image "$IMAGE" --region "$REGION" --allow-unauthenticated   --cpu=1 --memory=1Gi --max-instances=3   --set-env-vars MODEL_BUCKET=ml-image-clf,MODEL_PATH=model/model.pt,GOOGLE_CLOUD_PROJECT=${PROJ}
