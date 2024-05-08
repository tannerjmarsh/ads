import requests
import os

# get base url from environment
base_url = os.environ.get("API_SERVICE_URL", "http://ads-chat-api-service:9000")
base_url = "http://ads-chat-api-service:9000"

def create_session():
    url = base_url + "/new_session"
    response = requests.post(url)
    print(response)
    session_id = response.json()['session_id']
    return session_id

def chat(session_id: int, message: str):
    url = base_url + "/chat"
    response = requests.post("http://ads-chat-api-service:9000/chat", json={"session_id":5, "message": message})
    # response = requests.post(url, json={"session_id": session_id, "message": message})
    return response.json()

def reset(session_id: int):
    url = base_url + "/reset"
    response = requests.post(url, json={"session_id": session_id})

