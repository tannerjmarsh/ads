import os


from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd
from fastapi import File
from tempfile import TemporaryDirectory
from api import model
from api.utils import serialize_history, call_ads_api
from api.backup import BackupService, feedback_path
from pydantic import BaseModel
import api.model as model
import sys
import json
from langchain_core.exceptions import OutputParserException
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Initialize Backup Service
backup_service = BackupService()

# Initialize sessions
session_counter_file = "/app/persistent/session_counter.json"
sessions_lock = asyncio.Lock()
session_counter = 0
sessions = {}


# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")

# Initialize chat with all default parameters
chat = model.Chat()

class SessionID(BaseModel):
    session_id: int

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    print("Startup tasks!")    

    # Setup persistent folder
    os.makedirs(feedback_path, exist_ok=True)

    # Read session counter from the last stored value
    global session_counter
    if os.path.exists(session_counter_file):
        with open(session_counter_file, "r") as file:
            session_counter = json.load(file)['count']

    # Start the backup service
    # asyncio.create_task(tracker_service.track())

@app.on_event("shutdown")
def shutdown():
    print("Shutdown tasks!")
    print("Saving session counter...")
    with open(session_counter_file, "w") as file:
        json.dump({"count": session_counter}, file)

# Routes
@app.get("/")
async def get_index():
    return {"message": "Welcome to the API Service"}

@app.post("/new_session")
async def create_session():
    async with sessions_lock:
        global session_counter
        session_counter += 1
        session_id = session_counter
        sessions[session_id] = model.create_memory()
        print(f"all sessions: {sessions}", file=sys.stderr)
        return {"session_id": session_id}

@app.get("/sessions")
async def get_sessions():
    return sessions


@app.post("/delete_session")
async def delete_session(body: SessionID):
    session_id = body.session_id
    del sessions[session_id]


@app.post("/save_session_counter")
async def save_session_counter():
    print("Saving session counter...")
    with open(session_counter_file, "w") as file:
        json.dump({"count": session_counter}, file)

#TODO: Update this API. It is outdated
class ClientChatRequest(BaseModel):
    session_id: int
    message: str # rename this to something better
    k_examples: int = 3
    debug_context_examples: bool = False
    temperature: float = 0.0

class ChatResponse(BaseModel):
    session_id: int
    message: str
    examples_nl: list
    examples_solr: list
    ads_response: dict

@app.post("/chat")
async def get_completion(chat_request: ClientChatRequest):
    print(f"\n***chat request initiatied:***\n {chat_request}***\n", file=sys.stderr)
    session_id, message = chat_request.session_id, chat_request.message
    print(f"\n***{chat}***\n", file=sys.stderr)

    # get chat history for this session
    chat_memory = sessions[session_id]
    history = chat_memory.load_memory_variables({})['history']

    # get example retriever
    chat_retriever = chat.create_retriever(k=chat_request.k_examples)

    # retrieve examples
    examples = chat_retriever.get_relevant_documents(message)
    # examples = chat_retriever.invoke(message)
    examples_nl = [ex.page_content for ex in examples]
    examples_solr = [ex.metadata for ex in examples]
    for nl_text, solr in zip(examples_nl, examples_solr):
        print(f"NL: {nl_text}\nSOLR: {solr}\n", file=sys.stderr)

    # get chat completion
    chain_result = chat.chat_chain.invoke(
        {
            "input": message, 
            "history": history,
            "specific_examples": examples,
        }
    )
    raw_output = chain_result['output'].content
    print(f"unparsed chain output: {raw_output}\n", file=sys.stderr)

    # update the chat history
    chat_memory.save_context(
        {"input": chain_result["input"]}, 
        {"output": chain_result['output'].content}
    )

    # initialize response object
    chat_result = {
        "session_id": session_id, 
        "message": raw_output,
        "examples_nl": examples_nl,
        "examples_solr": examples_solr, 
        "context_examples": examples,
        "ads_response": {},
        # TODO: introduce a "status" field to indicate if the request was successful
    }

    # parse the json output
    parsed_output = None
    try:
        parsed_output = chat.output_parser.invoke(chain_result['output'].content)
        print(f"parsed chain output: {parsed_output}\n", file=sys.stderr)
    except OutputParserException as e:
        print(f"error parsing json: {e}\n", file=sys.stderr)
        corrected_output = chat.fixer_chain.invoke({"input": raw_output})
        parsed_output = chat.output_parser.invoke(corrected_output)
        print(f"corrected output: {corrected_output}\n", file=sys.stderr)

    if parsed_output:
        # made ads api request
        ads_response = call_ads_api(parsed_output)
        chat_result["ads_response"] = ads_response

    print(f"finished chat completion\n", file=sys.stderr)

    return chat_result

class TranslateRequest(BaseModel):
    nl_query: str
    k_examples: int = 3

@app.post("/translate")
async def translate(req: TranslateRequest):
    print(f"\n\nTRANSLATE ENDPOINT", file=sys.stderr)
    print(req, file=sys.stderr)

    # get example retriever
    chat_retriever = chat.create_retriever(k=req.k_examples)

    # retrieve examples
    examples = chat_retriever.get_relevant_documents(req.nl_query)
    print(examples, file=sys.stderr)

    # get chat completion
    chain_result = chat.chat_chain.invoke(
        {
            
            "input": req.nl_query, 
            "history": "",
            "specific_examples": examples,
        }
    )
    print("\n\t chain result before parsing:", file=sys.stderr)
    raw_output = chain_result['output'].content
    print(f"\t{raw_output}", file=sys.stderr)

    # parse the json output
    output = chat.output_parser.invoke(chain_result['output'].content)

    print("\n\t chain result after parsing:", file=sys.stderr)
    print(f"\t{output}", file=sys.stderr)

    # made ads api request
    ads_response = call_ads_api(output, bibcode=True)

    chat_result = {
        "chat_response": output, 
        "ads_response": ads_response
    }

    return chat_result

class NLToSolrRequest(BaseModel):
    nl_query: str
    n_icl: int = 3
    df_examples: str = ""


@app.post("/nl_to_solr")
async def nl_to_solr(req: NLToSolrRequest):
    print(f"\n\nNL TO SOLR ENDPOINT", file=sys.stderr)
    print(req, file=sys.stderr)

    if chat.icl_type == "random":
        df_examples = pd.read_json(req.df_examples)
        examples = []
        for _, row in df_examples.iterrows():
            examples.append(
                Document(page_content=row['nl'], metadata={'solr': row['solr']})
            )
        print(examples, file=sys.stderr)
    elif chat.icl_type == "rag":
        if req.n_icl >= 1:
            # get example retriever
            chat_retriever = chat.create_retriever(k=req.n_icl)

            # retrieve examples
            examples = chat_retriever.get_relevant_documents(req.nl_query)
            print(examples, file=sys.stderr)
        else:
            examples = ""

    # get chat completion
    chain_result = chat.chat_chain.invoke(
        {
            "input": req.nl_query, 
            "history": "",
            "specific_examples": examples,
        }
    )
    print("\n\t chain result before parsing:", file=sys.stderr)
    raw_output = chain_result['output'].content
    print(f"\t{raw_output}", file=sys.stderr)

    # parse the json output
    parsed_output = None
    try:
        parsed_output = chat.output_parser.invoke(raw_output)
        print(f"parsed chain output: {parsed_output}\n", file=sys.stderr)
    except OutputParserException as e:
        print(f"error parsing json: {e}\n", file=sys.stderr)
        corrected_output = chat.fixer_chain.invoke({"input": raw_output})
        parsed_output = chat.output_parser.invoke(corrected_output)
        print(f"corrected output: {corrected_output}\n", file=sys.stderr)

    chat_result = {
        "solr_response": parsed_output, 
    }

    return chat_result

class ChatParams(BaseModel):
    llm_model_name: str
    temperature: float
    embedding_model_name: str = "HF"
    icl_type: str
    fixed_icl: bool = False
    vector_database_type: str = "pinecone"

@app.post("/update_chat_params")
async def update_chat_params(req: ChatParams):
    print(f"\n\nUPDATE CHAT PARAMS ENDPOINT", file=sys.stderr)
    print(req, file=sys.stderr)

    global chat
    chat = model.Chat(
        llm_model_name=req.llm_model_name,
        temperature=req.temperature,
        embedding_model_name=req.embedding_model_name,
        icl_type=req.icl_type,
        fixed_icl=req.fixed_icl,
        vector_database_type=req.vector_database_type
    )
    # chat.set_llm_model(req.llm_model_name)
    # chat.set_temperature(req.temperature)
    # chat.set_embedding_model(req.embedding_model_name)

    return {"message": "Chat parameters updated"}

@app.get("/get_chat_params")
async def get_chat_params():
    print(f"\n\nGET CHAT PARAMS ENDPOINT", file=sys.stderr)

    chat_params = ChatParams(
        llm_model_name=chat.llm_model_name,
        temperature=chat.temperature,
        embedding_model_name=chat.embedding_model_name,
        fixed_icl=chat.fixed_icl
    )

    return chat_params.dict()

class FeedbackRequest(BaseModel):
    session_id: int
    positive: bool
    nl_request: str
    llm_solr: str
    human_solr: str

@app.post("/feedback")
async def feedback(body: FeedbackRequest):
    session_id = body.session_id
    file_path = os.path.join(feedback_path, f"{session_id}.json")
    print(os.getcwd(), file=sys.stderr)
    with open(file_path, "w") as f:
        json.dump(body.dict(), f)


