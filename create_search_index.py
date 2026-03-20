import logging
from src.azure_config import load_azure_config
from src.azure_indexer import get_index_client, create_or_update_index

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting index creation process...")
    
    # 1. Load config
    try:
        config = load_azure_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
        
    endpoint = config['SEARCH_ENDPOINT']
    api_key = config['SEARCH_API_KEY']
    index_name = config['SEARCH_INDEX_NAME']
    
    # 2. Get client
    logger.info("Authenticating with Azure AI Search...")
    index_client = get_index_client(endpoint, api_key)
    
    # 3. Create or update the index
    try:
        create_or_update_index(index_client, index_name)
        logger.info("Index setup complete.")
    except Exception as e:
        logger.error(f"Failed to create/update index: {e}")

if __name__ == "__main__":
    main()
