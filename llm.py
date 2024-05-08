import os
import requests
from urllib.parse import urlencode

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.vectorstores import Pinecone
from langchain.prompts import load_prompt
import pinecone
import download_ads


def get_pinecone_langchain_client(index_name: str, embedding, embedding_dim) -> Pinecone:
    # Initialize pinecone module
    pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
    )

    # Get or create pinecone index
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric="cosine",
            dimension=embedding_dim,
        )
    index = pinecone.Index(index_name)

    # Create LangChain pinecone client
    # note: embedding.embed_query is the function that's called to do the embedding
    pinecone_vectorstore = Pinecone(index=index, embedding=embedding, text_key="text")
    return pinecone_vectorstore

def update_memory(response):
    print(response.content)
    chat_memory.save_context(inputs, {"output": response.content})
    return response.content

def create_chain():

    # model
    model = ChatOpenAI()

    # output parser
    query_schema = ResponseSchema(
        name="q",
        description="The structured query based on Human input to be sent to ADS",
    )
    response_schemas = [query_schema]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()    

    # prompt
    template_path = '../../data/forward_templates/forward_prompt_simple_history.yaml'
    chat_prompt_template = load_prompt(template_path)
    partial_input_vars = {
        "fields": download_ads.get_fields_names(),
        "operators": download_ads.get_operator_names(),
        "multi_query_paragraph": download_ads.get_multi_query_paragraph(), 
        "fields_examples": str(download_ads.get_examples()),
        "operators_examples": str(download_ads.get_operators_info()["name_example_explanation"]),
        "explanation": "The AI should only output the answer and no additional information.",
        "format_instructions": format_instructions   
    }
    chat_partial_template = chat_prompt_template.partial(**partial_input_vars)

    # embedding
    embedding_choices = {
        "HF": {"model": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), "dim": 384},
        "OpenAI": {
            "model": OpenAIEmbeddings(),  # using this is annoying since rate limiting of 3/minute
            "dim": 1536,
        },
    }
    embedding = embedding_choices["HF"]

    # vectorstore
    pinecone_vectorstore = get_pinecone_langchain_client("demo", embedding=embedding["model"], embedding_dim=embedding["dim"])

    # retriever
    retriever = pinecone_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # memory
    chat_memory = ConversationBufferMemory(return_messages=True)
    chat_chain = (

    #chain
        RunnablePassthrough.assign(
            history = RunnableLambda(chat_memory.load_memory_variables) | itemgetter("history")
        )
        | RunnablePassthrough.assign(
            specific_examples = itemgetter("input") | retriever
        )
        | chat_partial_template
        # | RunnableLambda(print_prompt)
        | model
        | RunnableLambda(update_memory)
        | output_parser
    )

    return chat_chain


def create_model(
    include_explanation: bool = False, verbose: bool = False
) -> ConversationChain:
    prompt_template = create_prompt(include_explanation)

    llm = ChatOpenAI(temperature=0.0, openai_api_key=os.environ["OPENAI_API_KEY"])
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm, memory=memory, prompt=prompt_template, verbose=verbose
    )

    return conversation


def create_prompt(include_explanation: bool = False) -> ChatPromptTemplate:
    explanation = (
        "The AI should produce the output. The AI should also provide a step-by-step explanation of how it arrived at the output. It is important that the AI provides both the output and the explanation so a human can understand how it created that output."
        if include_explanation
        else "The AI should only output the answer and no additional information."
    )

    chain_of_thought = "Before providing the final answer, the AI should think through the problem step-by-step."

    query_schema = ResponseSchema(
        name="q",
        description="The structured query based on Human input to be sent to ADS",
    )
    response_schemas = [query_schema]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)

    format_instructions = output_parser.get_format_instructions()

    fields = download_ads.get_fields_names()

    operators = download_ads.get_operator_names()

    fields_examples = str(download_ads.get_examples())

    operators_examples = str(
        download_ads.get_operators_info()["name_example_explanation"]
    )

    specific_examples = """
Human: finds articles published between 1980 and 1990 by John Huchra
AI: ```json
{
"q": "author:\"Huchra, John\" year:1980-1990"
}
```
Human: What are papers that mention neural networks in the abstract?
AI: ```json
{
"q": "abs:\"neural networks\""
}
```
Human: Give me papers that mention neural networks in the title or keywords or abstract
AI: ```json
{
"q": "abs:\"neural networks\""
}
```
Human: Papers with that contain neural networks in the full text
AI: ```json
{
"q": "body:\"neural networks\""
}
Human: Everything from 2002 to 2008
AI: ```json
{
"q": "year:2002-2008"
}
```
Human: What papers by Kurtz, et al discuss weak lensing?
AI: ```json
{
"q": "author:\"Kurtz\" abs:\"weak lensing\""
}
```
Human: What papers by Alberto, et al discuss astronomy?
AI: ```json
{
"q": "author:\"Alberto\" abs:\"astronomy\""
}
```
Human: Show me papers about exoplanets with data from the MAST archive
AI: ```json
{
"q": "abs:\"exoplanets\" data:MAST"
}
```
Human: Give me papers which are like those from 2003AJ....125..525J
AI: ```json
{
"q": "similar(bibcode:2003AJ....125..525J)"
}
```
Human: Find me papers which can help me understand neural networks
AI: ```json
{
"q": "useful:\"neural networks\""
}
```
Human: return the top 100 most cited astronomy papers
AI: ```json
{
"q": "topn(100, database:astronomy, citation_count desc)"
}
```
Human: Give me the most important paper from Pavlos in 2020 100 most cited astronomy papers
AI: ```json
{
"q": "topn(1, author:\"Pavlos\", citation_count desc)"
}
```
Human: Give me the top 3 papers from Alberto between 2018 and 2022
Response: ```json
{
"q": "author:\"Alberto\" year:2018-2022",
"rows": 3,
"sort": "citation_count desc"
}
```

"""

    multi_query_paragraph = download_ads.get_multi_query_paragraph()

    template = """

INSTRUCTIONS: 

The following is a conversation between a human and an AI. The AI should answer the question based on the context, examples, and current conversation provided. If the AI does not know the answer to a question, it truthfully says it does not know. 

{chain_of_thought}


CONTEXT: 

The AI is an expert database search engineer. Specifically, the AI is trained to create structured queries that are submitted to NASA Astrophysics Data System (ADS), a digital library portal for researchers in astronomy and physics. The ADS system accepts queries using the Apache Solr search syntax. 

Here are all available fields and operators in the ADS database, where each field is separated by a space in this list: {fields} {operators}

{multi_query_paragraph}


AVAILABLE FIELDS: 

Here is an example for each of the available fields in the ADS database. The formatting is a Python list of lists. The inner list corresponds to an available field, is five elements long, and each element starts and ends with a single quote e.g. '. The first element is keywords associated with the field, the second element is the query syntax, the third element is the example query, the fourth element is associated notes, and the fifth element is the field name: 
{fields_examples}

AVAILABLE OPERATORS:
Here is an example for each of the available operators in the ADS database. The formatting is a Python list of lists. The inner list corresponds to an available operator, is three elements long, and each element starts and ends with a single quote e.g. '. The first element is the operator name, the second element is the example query, and the third element is associated notes: 
{operators_examples}


EXAMPLES:

The examples below are references for a typical, singular Human and AI interaction that provides the correct answer to a Human question.

{specific_examples}

The AI should create a similar query based on the question from the user.

{explanation}

{format_instructions}

Current conversation:\n{history}\nHuman: {input}\nAI:
"""

    prompt_template = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={
            "format_instructions": format_instructions,
            "fields": fields,
            "operators": operators,
            "fields_examples": fields_examples,
            "operators_examples": operators_examples,
            "specific_examples": specific_examples,  # could not pass this in directly to the PromptTemplate as I was getting a Pydantic ValidationError due to the brackets
            "explanation": explanation,
            "chain_of_thought": chain_of_thought,
            "multi_query_paragraph": multi_query_paragraph,
        },  # required to be either a string value or function that returns string values
        output_parser=output_parser,
    )

    return prompt_template


def call_ads_api(api_query):
    encoded_query = urlencode(api_query)

    results = requests.get(
        "https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query),
        headers={"Authorization": "Bearer " + os.environ["ADS_DEV_KEY"]},
    )

    return results.json()


if __name__ == "__main__":
    conversation = create_model()
    conversation.predict(
        input="finds articles published between 1980 and 1990 by John Huchra"
    )
    print(
        repr(conversation.memory.chat_memory.messages[-1].content)
    )  # IMPORTANT: this is what we really want to see for parsing the output (e.g. ran into an issue with not having double backslashes around the field that require strings e.g. author)
