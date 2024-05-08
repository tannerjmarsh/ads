IMAGE_NAME="ads-chat-frontend"
export NETWORK_NAME="chat-network"
export API_SERVICE_URL="http://ads-chat-api-service:9000"

docker network inspect ${NETWORK_NAME} >/dev/null 2>&1 || docker network create ${NETWORK_NAME}

docker build -t ${IMAGE_NAME} -f Dockerfile .

docker run --rm --name ${IMAGE_NAME} -ti \
--mount type=bind,source="$(pwd)",target=/app \
--network ${NETWORK_NAME} \
-p 8501:8501 \
-e OPENAI_API_KEY=${OPENAI_API_KEY} \
-e ADS_DEV_KEY=${ADS_DEV_KEY} \
-e API_SERVICE_URL=${API_SERVICE_URL} \
 ${IMAGE_NAME}
