import os

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_core.tools import tool

@tool
def calculator(a: float, b: float, type: str) -> float:
    """Performs mathematical calculations, addition, subtraction, multiplication, division, modulus.
    Args: 
        a (float): first float number
        b (float): second float number
        type (str): the type of calculation to perform, can be addition, subtraction, multiplication, division, modulus
    """

    if type == "addition":
        return a + b
    elif type == "subtraction":
        return a - b
    elif type == "multiplication":
        return a * b
    elif type == "division":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b
    elif type == "modulus":
        return a % b
    else:
        raise TypeError(f"{type} is not an option for type, choose one of addition, subtraction, multiplication, division, modulus")

@tool
def duck_web_search(query: str) -> str:
    """Use DuckDuckGo to search the web.

    Args:
        query: The search query.
    """
    search = DuckDuckGoSearchRun().invoke(query=query)
    
    return {"duckduckgo_web_search": search}

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    documents = WikipediaLoader(query=query, load_max_docs=3).load()
    processed_documents = "\n\n---\n\n".join(
        [
            f'Document title: {document.metadata.get("title", "")}. Summary: {document.metadata.get("summary", "")}. Documents details: {document.page_content}'
            for document in documents
        ])
    return {"wiki_results": processed_documents}

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    documents = ArxivLoader(query=query, load_max_docs=3).load()
    processed_documents = "\n\n---\n\n".join(
        [
            f'Document title: {document.metadata.get("title", "")}. Summary: {document.metadata.get("summary", "")}. Documents details: {document.page_content}'
            for document in documents
        ])
    return {"arxiv_results": processed_documents}

@tool
def tavily_web_search(query: str) -> str:
    """Search the web using Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    search_engine = TavilySearchResults(max_results=3)
    search_documents = search_engine.invoke(input=query)
    web_results = "\n\n---\n\n".join(
        [
            f'Document title: {document["title"]}. Contents: {document["content"]}. Relevance Score: {document["score"]}'
            for document in search_documents
        ])
    return {"web_results": web_results}