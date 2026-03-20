import logging
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

# Dimension for sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSIONS = 384

def get_index_client(endpoint: str, api_key: str) -> SearchIndexClient:
    """
    Establish a connection to the Azure AI Search Index client.
    
    Args:
        endpoint (str): The Azure Search resource endpoint.
        api_key (str): The API key for authentication.
        
    Returns:
        SearchIndexClient: The authenticated client.
    """
    credential = AzureKeyCredential(api_key)
    return SearchIndexClient(endpoint=endpoint, credential=credential)

def create_or_update_index(client: SearchIndexClient, index_name: str) -> None:
    """
    Define the schema and create or update the Azure Search index.
    
    Fields:
      - id: unique string key for Azure
      - doc_id: integer representing the document ID
      - title: searchable text
      - abstract_text: searchable text
      - text: searchable combined text (for keyword search)
      - content_vector: vector field for embeddings
      
    Args:
        client (SearchIndexClient): The search index client.
        index_name (str): The name of the index to create/update.
    """
    logger.info(f"Defining schema for index '{index_name}'...")
    
    # Define the fields required
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.Int32, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="abstract_text", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="my-vector-profile"
        )
    ]
    
    # Configure the vector search configuration
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw-config"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="my-vector-profile",
                algorithm_configuration_name="my-hnsw-config"
            )
        ]
    )
    
    # Create the index model
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    
    logger.info(f"Creating or updating index '{index_name}'...")
    result = client.create_or_update_index(index)
    logger.info(f"Index '{result.name}' successfully created or updated.")
