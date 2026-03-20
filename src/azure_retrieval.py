import logging
from typing import List, Dict, Any, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

logger = logging.getLogger(__name__)

def get_search_client(endpoint: str, api_key: str, index_name: str) -> SearchClient:
    """
    Establish a connection to the Azure AI Search Search component.
    
    Args:
        endpoint (str): The Azure Search resource endpoint.
        api_key (str): The API key for authentication.
        index_name (str): The name of the target index.
        
    Returns:
        SearchClient: The authenticated client.
    """
    credential = AzureKeyCredential(api_key)
    return SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

def format_results(results_iterator) -> List[Dict[str, Any]]:
    """Format the results into a clean list of dictionaries."""
    formatted = []
    for result in results_iterator:
        # Some searches provide '@search.score' (keyword/hybrid), others yield '@search.score' differently.
        score = result.get('@search.score', 0.0)
        formatted.append({
            'doc_id': result.get('doc_id'),
            'title': result.get('title'),
            'score': score
        })
    return formatted

def run_keyword_search(client: SearchClient, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a keyword search using BM25 across standard text fields.
    """
    logger.info(f"Running keyword search for: '{query_text}'")
    # Searchable fields: title, abstract_text, text. We use text (the combined one) natively.
    results = client.search(
        search_text=query_text,
        top=top_k,
        select=['doc_id', 'title']
    )
    return format_results(results)

def run_vector_search(client: SearchClient, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a pure vector search against the content_vector field.
    """
    logger.info(f"Running purely vector search to retrieve top {top_k} results")
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )
    
    results = client.search(
        search_text=None,
        vector_queries=[vector_query],
        top=top_k,
        select=['doc_id', 'title']
    )
    return format_results(results)

def run_hybrid_search(
    client: SearchClient, 
    query_text: str, 
    query_vector: List[float], 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform a hybrid search combining keyword search and vector similarity.
    """
    logger.info(f"Running hybrid search for: '{query_text}'")
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )
    
    results = client.search(
        search_text=query_text,
        vector_queries=[vector_query],
        top=top_k,
        select=['doc_id', 'title']
    )
    return format_results(results)
