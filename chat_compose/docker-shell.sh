#!/bin/bash

set -e

export DEV=1
export PUSH_IMAGE=0
export API_SERVICE_IMAGE_NAME="ads-chat-api-service"
export FRONTEND_IMAGE_NAME="ads-chat-frontend"


# Create the network if we don't have it yet
docker network inspect chat-network-compose >/dev/null 2>&1 || docker network create chat-network-compose


if [[ "${DEV}" == "1" ]]; then
    # Build the image based on the Dockerfile
    docker build -t ${FRONTEND_IMAGE_NAME} -f ../chat_streamlit/Dockerfile ../chat_streamlit
    docker build -t ${API_SERVICE_IMAGE_NAME} -f ../chat_api/Dockerfile ../chat_api


    if [[ "${PUSH_IMAGE}" == "1" ]]; then
        echo "Pushing image to Docker Hub"
    fi
else
    echo "Pulling image from Docker Hub"
fi

 
# Run All Containers
# docker compose run --rm --service-ports ads-chat-api-service
docker compose up