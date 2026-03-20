import os
import logging
from dotenv import load_dotenv

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_azure_config() -> dict:
    """
    Load Azure settings from .env and validate required environment variables.
    
    Returns:
        dict: A dictionary containing the Azure configuration.
    """
    # Load .env file if present (local dev). In Azure, system environment variables will be used directly.
    load_dotenv()
    
    config = {
        "SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "SEARCH_API_KEY": os.getenv("AZURE_SEARCH_API_KEY"),
        "SEARCH_INDEX_NAME": os.getenv("AZURE_SEARCH_INDEX_NAME"),
        # We also have AZURE_STORAGE_CONNECTION_STRING as per instructions, though maybe not strictly needed for just Search right now.
        "STORAGE_CONNECTION_STRING": os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    }
    
    # Validate required variables for AI Search integration
    missing_vars = [k for k, v in config.items() if not v and k != "STORAGE_CONNECTION_STRING"]
    
    if missing_vars:
        error_msg = f"Missing required Azure settings: {', '.join(missing_vars)}. Please provide them via local .env or Cloud Environment Variables/Secrets."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    return config
