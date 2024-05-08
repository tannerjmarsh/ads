from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import requests
import streamlit as st
# from langsmith import Client
import query as api
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import sys
# client = Client()


# for key in st.session_state.keys():
#     del st.session_state[key]

st.set_page_config(page_title="ADS Chat", page_icon="🔎")
st.title("🔎 ADS Chat")
button_css =""".stButton>button {
    color: #4F8BF9;
    border-radius: 50%;
    height: 2em;
    width: 2em;
    font-size: 4px;
}"""
st.markdown(f'<style>{button_css}</style>', unsafe_allow_html=True)

# st.button("test", key="test_button")
# st.write(st.session_state)

def parse_feedback(positive: bool):
    if positive:
        st.session_state.messages.append(HumanMessage(content="👍"))
        submit_feedback(positive=True)
    else:
        st.session_state.messages.append(HumanMessage(content="👎"))
        st.session_state.solr_correction_needed = True

def submit_feedback(positive: bool):
    # submit feedback to api service endpoint
    body = {
        "session_id": st.session_state["session_id"],
        "positive": positive,
        "nl_request": st.session_state["last_request"],
        "llm_solr": st.session_state["last_response"]["ads_response"]["responseHeader"]["params"]["q"],
        "human_solr": "" if positive else st.session_state["solr_correction"],
    }
    print(body)
    response = requests.post(
        "http://ads-chat-api-service:9000/feedback",
        json=body
    )

def chat_completion(prompt):
    request = {
        "session_id":st.session_state["session_id"], 
        "message": prompt,
        "k_examples": st.session_state["num_context_examples"],
        "debug_context_examples": True
    }
    response = requests.post(
        "http://ads-chat-api-service:9000/chat", 
        json=request
    )
    print(f"\nchat_completion: raw response: {response}\n", file=sys.stderr)
    response = response.json()
    st.session_state.last_response = response
    
    chat_response = response["message"]
    ads_response = response["ads_response"]
    examples_nl = response["examples_nl"]
    examples_solr = response["examples_solr"]


    output = f""

    if not ads_response:
        return chat_response
        

    q_param = response["ads_response"]["responseHeader"]["params"]["q"]
    sort_param = response["ads_response"]["responseHeader"]["params"]["sort"]
    fl_param = response["ads_response"]["responseHeader"]["params"]["fl"]

    output += f"`q={q_param}`  \n"
    output += f"`fl={fl_param}`  \n"
    output += f"`sort={sort_param}`  \n"

    # error checking
    if response["ads_response"]["responseHeader"]["status"] != 0:
        error = response["ads_response"]["error"]
        output += f"Error:  \n "
        output += f"`{error}`  \n"
    else:
        num_found = response["ads_response"]["response"]["numFound"]
        titles = [doc["title"][0] for doc in response["ads_response"]["response"]["docs"]]
        years = [doc["year"] for doc in response["ads_response"]["response"]["docs"]]
        authors = [doc["author"][0:max(3,len(doc["author"]))] for doc in response["ads_response"]["response"]["docs"]]

        output += f"matches found: {num_found}  \n"
        output += " ".join(["- " + title + "  \n" + year + "  \n"  for title, year in zip(titles, years)])

    return output



if "first_query" not in st.session_state:
    st.session_state["first_query"] = True
if "show_feedback" not in st.session_state:
    st.session_state["show_feedback"] = False
if "session_id" not in st.session_state:
    st.session_state["session_id"] = api.create_session()
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(
        content="Welcome! Please enter your ADS search.   \
            \n\nYou can also restart the chat anytime by typing `/reset`  \
            \n\n You can also see the full json response from the server by typing `/response`  \
            \n\n See the context examples by typing `/examples`"

    )]
if "solr_correction_needed" not in st.session_state:
    st.session_state["solr_correction_needed"] = False

# sidebar
with st.sidebar:
    st.header("Debug panel")

    # st.subheader("Fields")
    # st.write("**sort:**")
    # st.write("**fl:**")


    st.subheader("Examples")
    number = st.number_input("Number of context examples", step=1, min_value=0, max_value=40, value=3, key="num_context_examples")



# print the chat messages and feedback
for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    else:
        st.chat_message("assistant").write(msg)
if st.session_state["show_feedback"]:
    st.session_state["show_feedback"] = False
    col_blank, col_text, col1, col2 = st.columns([10, 2,1,1])
    with col_text:
        st.text("Feedback:")

    with col1:
        st.button("👍", on_click=parse_feedback, args=(True,))

    with col2:
        st.button("👎", on_click=parse_feedback, args=(False,))

if st.session_state["solr_correction_needed"]:
    st.session_state["solr_correction_needed"] = False
    st.text_input(
        "solr correction", 
        value=st.session_state["last_response"]["ads_response"]["responseHeader"]["params"]["q"],
        on_change=submit_feedback,
        args=(False,),
        key="solr_correction",
    )
elif prompt := st.chat_input(key="input_message"):

    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))


    if prompt == "/reset":
        print(st.session_state["session_id"])
        # reset the session on the server
        response = requests.post(f"http://ads-chat-api-service:9000/delete_session", 
                                json={"session_id":st.session_state["session_id"]})
        # empty the session state
        for key in st.session_state.keys():
            del st.session_state[key]
        # rerun the script
        st.rerun()
    elif prompt == "/response": #refactor this
        response_message = "No searches yet."
        if "last_response" in st.session_state:
            st.session_state.messages.append(st.session_state["last_response"])
            response_message = st.session_state["last_response"]
        with st.chat_message("assistant"):
            st.write(response_message)
    elif prompt == "/examples":
        response_message = "No searches yet."
        if "last_response" in st.session_state:
            response_message = " ".join(["- " + example["page_content"] + "  \n" + example["metadata"]["solr"] + "  \n\n"
                                         for example in st.session_state["last_response"]["context_examples"]])
            st.session_state.messages.append(response_message)
        with st.chat_message("assistant"):
            st.write(response_message)
    elif prompt == "/sessions":
        response = requests.get(f"http://ads-chat-api-service:9000/sessions/")
        st.write(response.json())
    else:
        st.session_state["last_request"] = prompt

        with st.chat_message("assistant"):
            output = chat_completion(prompt)
            st.write(output)
            st.session_state.messages.append(AIMessage(content=output))

        if st.session_state["first_query"]:
            st.session_state["first_query"] = False
            st.session_state["show_feedback"] = True
            st.rerun()