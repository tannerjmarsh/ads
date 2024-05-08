from collections import defaultdict
import re

import requests
from bs4 import BeautifulSoup


def get_examples() -> list[str, str, str, str, str]:
    """Getting all provided examples from Available Fields section on ADS' Search Syntax webpage: https://ui.adsabs.harvard.edu/help/search/search-syntax

    The formatting is a Python list of lists. The inner list corresponds to an available field, is five elements long, and each element starts and ends with a single quote e.g. '. The first element is keywords associated with the field, the second element is the query syntax, the third element is the example query, the fourth element is associated notes, and the fifth element is the field name

    Here is an example field output: ['Abstract/Title/Keywords', 'abs:“phrase”', 'abs:“dark energy”', 'search for word or phrase in abstract, title and keywords', 'abs']
    """
    return get_available_fields_info()["body"]


def get_fields_names() -> str:
    """Getting a list of all the fields from the ADS API, and then formatting to a string for the LLM prompt (partial_variables must be a string)"""
    data = get_available_fields_info()

    fields = get_fields_unformatted(data)

    return format_fields(fields)


def get_available_fields_info():
    """Getting the table, and pulling out a dictionary of lists based on the header and body

    Here is an example field: ['Abstract/Title/Keywords', 'abs:“phrase”', 'abs:“dark energy”', 'search for word or phrase in abstract, title and keywords']
    """
    response = requests.get("https://ui.adsabs.harvard.edu/help/search/search-syntax")
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the h3 element with the id attribute set to "available-fields"
    h3 = soup.find("h3", {"id": "available-fields"})

    # Find the table element that is immediately after the h3 element
    table = h3.find_next("table")

    header_elements = table.find("thead").find_all("th")

    header_text = [th.text for th in header_elements]

    data = defaultdict(list)
    data["header"] = header_text

    rows = table.find("tbody").find_all("tr")

    for row in rows:
        row_data = row.find_all("td")
        row_text = [r.text for r in row_data]
        data["body"].append(row_text)

    return data


def get_fields_unformatted(data: dict[str, list[str]]) -> list[str]:
    # e.g. ['Abstract/Title/Keywords', 'abs:“phrase”', 'abs:“dark energy”', 'search for word or phrase in abstract, title and keywords'], then subsetting to 'abs:“phrase”', then finally extracting 'abs'

    fields = []
    for row in data["body"]:
        if row[0] != "First Author":
            fields.append(row[1].split(":")[0])
        else:  # First Author is edge case
            fields.append("^")

    return fields


def format_fields(fields: list[str]) -> str:
    """Formatting the list of fields to a string for the LLM prompt

    E.g. if starting with a list of strings that looks like ['abs', 'abstract', 'ack'], we'll end up with 'abs abstract ack'
    """
    str_fields = str(fields)

    pattern = re.compile(r"'|,|\[|\]")
    return pattern.sub("", str_fields)


def get_multi_query_paragraph() -> str:
    response = requests.get("https://ui.adsabs.harvard.edu/help/search/search-syntax")
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the h3 element with the id attribute set to "combining-search-terms-to-make-a-compound-query"
    h3 = soup.find("h3", {"id": "combining-search-terms-to-make-a-compound-query"})

    return h3.find_next("p").text.split(" Some examples:")[0]


def get_operators_info() -> dict[str, list[str]]:
    """Getting the table, and pulling out a dictionary of lists based on the body"""
    response = requests.get(
        "https://ui.adsabs.harvard.edu/help/search/comprehensive-solr-term-list"
    )
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the h3 element with the id attribute set to "combining-search-terms-to-make-a-compound-query"
    article = soup.find("article", class_="post-content")

    operators_table = article.find_next("table").find_next("table")

    rows = operators_table.find("tbody").find_all("tr")

    data = defaultdict(list)

    for row in rows:
        row_data = row.find_all("td")
        row_text = [
            r.text.strip() if i != 0 else r.text.replace("()", "").strip()
            for i, r in enumerate(row_data)
        ]
        data["name_example_explanation"].append(row_text)

    data["names"] = [
        operator_name for operator_name, _, _ in data["name_example_explanation"]
    ]

    return data


def get_operator_names() -> list[str]:
    return format_fields(get_operators_info()["names"])


if __name__ == "__main__":
    print(str(get_operators_info()["name_example_explanation"]))
