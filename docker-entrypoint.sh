#!/bin/bash

echo "Container is running!!!"

# load secrets files into the environment
OPENAI_API_KEY_FILE="/run/secrets/open_ai_key"
ADS_KEY_FILE="/run/secrets/ads_key"
PINECONE_KEY_FILE="/run/secrets/pinecone_key"


if [ -f "$OPENAI_API_KEY_FILE" ]; then
    export OPENAI_API_KEY=$(cat "$OPENAI_API_KEY_FILE")
fi
if [ -f "$ADS_KEY_FILE" ]; then
    export ADS_DEV_KEY=$(cat "$ADS_KEY_FILE")
fi
if [ -f "$PINECONE_KEY_FILE" ]; then
    export PINECONE_API_KEY=$(cat "$PINECONE_KEY_FILE")
fi

# this will run the api/service.py file with the instantiated app FastAPI
uvicorn_server() {
    # uvicorn api.service:app --host 0.0.0.0 --port 9000 --log-level debug --reload --reload-dir api/ "$@"
    pipenv run uvicorn api.service:app --host 0.0.0.0 --port 9000 --log-level debug --reload --reload-dir api/ "$@"

}

uvicorn_server_production() {
    pipenv run uvicorn api.service:app --host 0.0.0.0 --port 9000 --lifespan on
}

export -f uvicorn_server
export -f uvicorn_server_production

echo -en "\033[92m
The following commands are available:
    uvicorn_server
        Run the Uvicorn Server
\033[0m
"

if [ "${DEV}" = 1 ]; then
  uvicorn_server
#   pipenv shell
else
  uvicorn_server_production
fi
