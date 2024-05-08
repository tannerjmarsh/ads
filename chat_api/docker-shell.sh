#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="ads-chat-api-service"
export BASE_DIR=$(pwd)
export NETWORK_NAME="chat-network"
OPEN_AI_KEY_FILE="../../secrets/open_ai_key.txt"
ADS_KEY_FILE="../../secrets/ads_key.txt"

export ADS_DEV_KEY=$(cat "$ADS_KEY_FILE")
export OPENAI_API_KEY=$(cat "$OPEN_AI_KEY_FILE")

docker network inspect ${NETWORK_NAME} >/dev/null 2>&1 || docker network create ${NETWORK_NAME}

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
--network ${NETWORK_NAME} \
-v "$BASE_DIR":/app \
-p 9000:9000 \
-e OPENAI_API_KEY=${OPENAI_API_KEY} \
-e ADS_DEV_KEY=${ADS_DEV_KEY} \
-e DEV=1 \
$IMAGE_NAME

