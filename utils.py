from urllib.parse import urlencode
import requests
import time
import os
import json
from json.decoder import JSONDecodeError
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import pandas as pd
from tqdm import tqdm

def get_bibcodes(solr_query: str) -> list[list[str]]:
    api_query = {}
    api_query['q'] = solr_query
    api_query['rows'] = 2000 
    api_query['fl'] = 'bibcode'
    api_query['start'] = 0

    
    bibcodes = []

    encoded_query = urlencode(api_query)
    results = requests.get(
        "https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query),
        headers={"Authorization": "Bearer " + os.environ["ADS_DEV_KEY"]},
    ).json()
    if 'response' not in results:
        #TODO: deal with errors related to objects
        print(f"Error with query: {solr_query}")
        print(results)
        return []

    num_found = results['response']['numFound']
    print(solr_query)
    print(f"total found: {num_found}")

    bibcodes.extend([result['bibcode'] for result in results['response']['docs']])

    while results['response']['numFound'] > (api_query['start'] + api_query['rows']):
        api_query['start'] += api_query['rows']
        encoded_query = urlencode(api_query)
        results = requests.get(
            "https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query),
            headers={"Authorization": "Bearer " + os.environ["ADS_DEV_KEY"]},
        ).json()
        bibcodes.extend([result['bibcode'] for result in results['response']['docs']])
        time.sleep(0.1)
        print(f"batch done {api_query['start']} - {api_query['start'] + api_query['rows']}")

        if api_query['start'] >= 50000:
            break

    # assert len(bibcodes) == num_found or len(bibcodes) == 50000, f"Number of bibcodes ({len(bibcodes)}) does not match number of results ({num_found})"
    return bibcodes

def batch_get_bibcodes(solr_queries: list[str]) -> list[list[str]]:
    bibcodes_list = []
    for solr_query in tqdm(solr_queries):
        bibcodes = get_bibcodes(solr_query)
        time.sleep(0.1)
        bibcodes_list.append(bibcodes)
    return bibcodes_list

def set_experiment_params(llm_model_name: str, temperature: float, embedding_model_name: str, icl_type: str, fixed_icl: bool, vector_database_type:str):
    """This function calls the update_chat_params endpoint to set the experiment parameters (which language model to use, whether to use fixed icl examples, etc.) in the api service"""
    url = "http://localhost:9099/update_chat_params"
    body = {
        "llm_model_name": llm_model_name,
        "temperature": temperature,
        "embedding_model_name": embedding_model_name,
        "icl_type": icl_type,
        "fixed_icl": fixed_icl, # TODO: remove this and use the experiment field instead
        "vector_database_type": vector_database_type
        # "reset_db": reset_db,
    }
    response = requests.post(url, json=body).json()
    #TODO: add assert to verify that params were actually updated
    return response

def batch_nl_to_solr(nl_queries, n_icl=3, model="gpt-3.5", embed_model="HF", temperature=0.0,fixed_icl=False, df_examples=None):
    """Note: some of the arguments are redundant. These parameters get set by set_experiment_params"""
    print(f"Batching NL to Solr with the following parameters: n_icl={n_icl}")
    url = "http://localhost:9099/nl_to_solr"
    solr_queries = []
    df_examples_str = df_examples.to_json(orient="records") if df_examples is not None else ""
    for nl_query in tqdm(nl_queries):
        try:
            body = {
                "nl_query": nl_query,
                "n_icl": n_icl,
                "df_examples": df_examples_str,
            }
            response = requests.post(url, json=body).json()
        except JSONDecodeError:            
            solr_queries.append(None)
            continue

        solr_queries.append(response["solr_response"]["q"])
        # bibcode = [paper["bibcode"] for paper in response["ads_response"]["response"]["docs"]]
        # bibcodes_list.append(bibcode)
    return solr_queries





def get_langchain_pinecone_client(embedding_model: str) -> Pinecone:
    embedding_choices = {
        "HF": {
            "model": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), 
            "dim": 384},
        "OpenAI": {
            "model": OpenAIEmbeddings(),  # using this is annoying since rate limiting of 3/minute
            "dim": 1536,
        },
    }
    embedding = embedding_choices[embedding_model]

    # Get or create pinecone index
    index_name = "icl-examples"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric="cosine",
            dimension=embedding["dim"],
        )
    index = pinecone.Index(index_name)

    # Create LangChain pinecone client
    lc_pinecone_client = Pinecone(index=index, embedding=embedding, text_key="text")
    return lc_pinecone_client


def embed_dataframe(df: pd.DataFrame, lc_pinecone_client: Pinecone):
    """This function takes a dataframe and a vectorstore and embeds the examples in the dataframe. The function assumes that the dataframe has the following structure:"""

    documents = []
    for _, row in df.iterrows():
        documents.append(
            Document(page_content=row['nl'], metadata={'solr': row['solr']})
        )

    print(lc_pinecone_client)
    print(documents)
    lc_pinecone_client.add_documents(documents)


def filter_experiments(directory: str, **filters):
    """
        # Example Usage:
        directory_path = "path/to/your/results/directory"
        filters = {
            "n_icl": 8,  # k value
            "model": "someModel"
        }
        matching_files, file_details = filter_experiments(directory_path, **filters)
        print(file_details)
    """
    files = os.listdir(directory)
    matching_files = []
    file_details = []

    for file in files:
        if "meta.json" in file:
            full_path = os.path.join(directory, file)
            with open(full_path) as f:
                details = json.load(f)
                if all(str(details.get(key)) == str(value) for key, value in filters.items()):
                    full_path = os.path.join(directory, file)
                    matching_files.append(full_path)
                    file_details.append(details)

    return matching_files, file_details

# def aggregate_results(results_directory: str, filters: dict, params: list[str]):
#     matching_files, file_details = filter_experiments(results_directory, **filters)

def sample_examples(df: pd.DataFrame, n: int) -> list[Document]:
    """This function samples n examples from the train dataset and returns a list of those examples represented as langchain Document objects. The function assumes that the dataframe has columns 'nl' and 'solr'."""
    df_sampled = df.sample(n)
    documents = []
    for _, row in df_sampled.iterrows():
        documents.append(
            Document(page_content=row['nl'], metadata={'solr': row['solr']})
        )
    return documents

def deserialize_pretty_json(f: str) -> str:
    with open(f, "r") as file:
        json_content = file.read()

    # Since the JSON objects might not be in a single line format, splitting by '}' and adding it back for proper JSON formatting
    json_objects = [json.loads(obj + '}') for obj in json_content.split('}') if obj.strip()]

    # Create DataFrame from list of JSON objects
    data_df = pd.DataFrame(json_objects)

    # Display the DataFrame to check it
    return data_df

def aggregate_results(results_directory: str, filters: dict, params: list[str]) -> pd.DataFrame:
    matching_files, file_details = filter_experiments(results_directory, **filters)
    """
    # Example Usage:
    # results_directory = 'results'
    # filters = {
    #     'dataset': 'kelly',
    #     'model': 'gpt-3.5-turbo-1106',
    # }
    # params = ['n_icl', 'jaccard']
    # df = aggregate_results(results_directory, filters, params)
    """
    
    matching_files, file_details = filter_experiments(results_directory, **filters)

    if len(matching_files) == 0:
        raise ValueError("No matching files found")

    df = pd.DataFrame(file_details)

    # Select only the columns specified in the 'params' list
    df_expanded = df['jaccard'].apply(pd.Series)
    result = pd.concat([df.drop("jaccard", axis=1), df_expanded], axis=1)
    return result[params]