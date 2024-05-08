
import os
import sys
from operator import itemgetter
import pinecone

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings

import api.download_ads as download_ads

class Chat:
    """A class that encapsulates of all the resources involved in running the chat model. This
    is useful because it makes it easy to create chat chains with different profiles
    (different llm, different embedding models, etc.).

    The components include:
    - model
    - output parser
    - output parser fixer (when output parser fails)
    - chat prompt
    - pinecone client
    - chat chain (the chain of operations to run the chat model)
    """
    def __init__(
            self, 
            llm_model_name: str = "gpt-3.5-turbo-1106", 
            embedding_model_name: str = "HF",
            temperature=0.0,
            prompt_template_path: str = "forward_prompt_simple_history.yaml",
            icl_type: str = "rag",
            fixed_icl: bool = False,
            vector_database_type: str = "pinecone"
        ):
        self.llm_model_name = llm_model_name
        self.llm = ChatOpenAI(
            model=llm_model_name,
            temperature=temperature
        )
        self.embedding_model_name = embedding_model_name
        self.temperature = temperature
        self.prompt_template_path = prompt_template_path
        self.icl_type = icl_type
        self.output_parser = self.__create_output_parser(self.llm)
        if self.icl_type == "fixed":
            self.chat_prompt = self.__create_prompt_fixed_icl()
        else:
            self.chat_prompt = self.__create_prompt(
                template_path=prompt_template_path,
                output_parser=self.output_parser
            )

        if vector_database_type == "pinecone":
            self.vectorstore = self.__create_vectorstore(embedding_model_name)
        # elif vector_database_type == "chroma":
        #     self.vectorstore = self.__create_chroma_vectorstore(embedding_model_name)
        else:
            raise ValueError("Invalid vector database type. Must be either 'pinecone' or 'chroma'.")

        self.chat_chain = self.__create_chain(self.chat_prompt, self.llm)
        self.fixer_chain = self.__create_parser_fixer_chain(self.llm)
        self.vector_database_type = vector_database_type
        

    def __create_output_parser(self, model):
        # generate format instructions for the prompt
        query_schema = ResponseSchema(
            name="q",
            description="The structured query based on Human input to be sent to ADS.",
        )
        sort_schema = ResponseSchema(
            name="sort",
            description="The sort order for the query, if specified in the Human input, otherwise 'date desc' if not specified",
        )
        fields_schema = ResponseSchema(
            name="fl",
            description="The fields to return, if specified in the Human input, otherwise 'author,title,year' if not specified",
        )
        object_schema = ResponseSchema(
            name="object",
            description="The astronomical object, if specified in the Human input, otherwise '' if not specified.",
        )
        response_schemas = [query_schema, sort_schema, fields_schema, object_schema]
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        fixing_parser = OutputFixingParser.from_llm(llm=model, parser=output_parser)
        return fixing_parser

    def __create_parser_fixer_chain(self, model, system_message: str = None):
        if system_message is None:
            system_message = """
            You are a helpful chatbot. The Human will provide you with a JSON object. Your purpose is to make any corrections so that the JSON string is valid. Only output the corrected JSON string.
            """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "{input}"),
            ]
        )
        chain = (
            prompt
            | self.llm
        )
        return chain

    def __create_prompt(self, template_path: str, output_parser: StructuredOutputParser):
        """Create a prompt for the chatbot to use"""
        format_instructions = output_parser.get_format_instructions()

        # generate partial input variables for the prompt
        partial_input_vars = {
            "fields": download_ads.get_fields_names(),
            "operators": download_ads.get_operator_names(),
            "multi_query_paragraph": download_ads.get_multi_query_paragraph(),
            "fields_examples": str(download_ads.get_examples()),
            "operators_examples": str(download_ads.get_operators_info()["name_example_explanation"]),
            "explanation": "The AI should only output the answer and no additional information.",
            "format_instructions": format_instructions
        }

        # Load promp template from a file. It is a PromptTemplate
        chat_prompt_template = load_prompt(template_path)

        # parially parameterize all input variables except for history, examples, and input
        chat_partial_template = chat_prompt_template.partial(**partial_input_vars)

        return chat_partial_template

    def __create_chain(self, chat_prompt, model):
        chat_chain = (
            RunnablePassthrough.assign(
                prompt=chat_prompt
            )
            | RunnablePassthrough.assign(
                output=itemgetter("prompt") | model
            )
        )
        return chat_chain

    def __create_pinecone_langchain_client(self, index_name: str, embedding, embedding_dim, reset: bool = False) -> Pinecone:
        # Initialize pinecone module
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
        )

        index = pinecone.Index(index_name)
        pinecone_vectorstore = Pinecone(index=index, embedding=embedding, text_key="text")

        return pinecone_vectorstore
    
    def __format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def __create_vectorstore(self, embedding_model_name: str = "HF"):
        embedding_choices = {
            "HF": {"model": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), "dim": 384},
            "OpenAI": {
                "model": OpenAIEmbeddings(),
                "dim": 1536,
            },
        }

        embedding = embedding_choices[embedding_model_name]

        pinecone_vectorstore = self.__create_pinecone_langchain_client(
            index_name="ads", 
            embedding=embedding["model"], 
            embedding_dim=embedding["dim"], 
            reset=False
        )

        return pinecone_vectorstore

    def __create_prompt_fixed_icl(include_explanation: bool = False) -> ChatPromptTemplate:
        """This is legacy code that needs to be cleaned up"""
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
                # "chain_of_thought": chain_of_thought,
                "multi_query_paragraph": multi_query_paragraph,
            },  # required to be either a string value or function that returns string values
            output_parser=output_parser,
        )

        return prompt_template

    def create_retriever(self, k: int = 3):
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        print(retriever)
        return retriever
    
    def set_llm_model(self, llm_model_name: str):
        self.llm_model_name = llm_model_name
        self.llm = ChatOpenAI(
            model=llm_model_name,
            temperature=self.temperature
        )
        self.chat_chain = self.__create_chain(self.chat_prompt, self.llm)
        self.output_parser = self.__create_output_parser(self.llm)
        self.fixer_chain = self.__create_parser_fixer_chain(self.llm)
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=self.llm_model_name,
            temperature=temperature
        )
        self.chat_chain = self.__create_chain(self.chat_prompt, self.llm)
        self.output_parser = self.__create_output_parser(self.llm)
        self.fixer_chain = self.__create_parser_fixer_chain(self.llm)
    

def create_memory():
    chat_memory = ConversationBufferMemory(return_messages=True)
    return chat_memory

def update_memory(memory, chain_result):
    memory.save_context(
        {"input": chain_result["input"]}, 
        {"output": chain_result["output"].content})