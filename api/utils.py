import os
import requests
from urllib.parse import urlencode

def serialize_history(memory: str):
    return f"session_id is {memory}"

def new_chat_memory():
    return "new_chat_memory"

def call_ads_api(api_query, bibcode=False):
    # api_query['rows'] = 10 if not api_query['rows'] else api_query['rows']
    # api_query['fl'] = 'author,title,year' if not api_query['fl'] else api_query['fl']
    # api_query['sort'] = 'date desc' if not api_query['sort'] else api_query['sort']

    # include bibcode in the field list if requested
    if bibcode:
        api_query['fl'] = ',bibcode'

    encoded_query = urlencode(api_query)


    results = requests.get(
        "https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query),
        headers={"Authorization": "Bearer " + os.environ["ADS_DEV_KEY"]},
    )

    return results.json()