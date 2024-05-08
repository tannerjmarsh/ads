FROM python:3.11-bullseye


RUN pip install --no-cache-dir --upgrade pip && \
    pip install beautifulsoup4 langchain openai streamlit streamlit-extras streamlit-chat requests

COPY . /app

WORKDIR /app

EXPOSE 8501

# HEALTHCHECK instruction tells Docker how to test a container to check that it is still working. Your container needs to listen to Streamlit’s (default) port 8501
# source: https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["/bin/bash","./docker-entrypoint.sh"]