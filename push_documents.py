import argparse
import logging
import os
from typing import List, Dict, Any

# Bypass tensorflow import bug from transformers
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_loader import load_corpus
from src.azure_config import load_azure_config
from src.azure_retrieval import get_search_client

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Free tier safe batch size for Azure uploads
UPLOAD_BATCH_SIZE = 100

def get_optimal_device() -> str:
    """Detect the optimal PyTorch device to use."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = "cpu"
        logger.info("CUDA not found. Falling back to CPU.")
    return device

def prepare_documents(corpus_df, model: SentenceTransformer, encode_batch_size: int) -> List[Dict[str, Any]]:
    """
    Format the documents for Azure AI Search, generating vector embeddings locally.
    
    Args:
        corpus_df (pd.DataFrame): The loaded corpus DataFrame.
        model (SentenceTransformer): Model for generating embeddings.
        encode_batch_size (int): Batch size for generating embeddings.
        
    Returns:
        List[Dict[str, Any]]: List of documents ready for Azure Search upload.
    """
    documents = []
    
    # Extract texts for batch embedding
    texts_to_embed = corpus_df['text'].tolist()
    logger.info(f"Generating embeddings for {len(texts_to_embed)} documents locally...")
    logger.info(f"Using encode batch size of {encode_batch_size}...")
    
    # Generate embeddings utilizing the specified device and batch processing
    embeddings = model.encode(
        texts_to_embed, 
        batch_size=encode_batch_size, 
        normalize_embeddings=True, 
        show_progress_bar=True
    )
    
    # Convert np arrays to lists for JSON serialization
    embeddings = [embedding.tolist() for embedding in embeddings]
    
    for idx, row in corpus_df.iterrows():
        doc = {
            "id": str(row['doc_id']),
            "doc_id": int(row['doc_id']),
            "title": str(row.get('title', '')),
            "abstract_text": str(row.get('abstract_text', '')),
            "text": str(row['text']),
            "content_vector": embeddings[idx]
        }
        documents.append(doc)
        
    return documents

def upload_in_batches(client, documents: List[Dict[str, Any]]):
    """
    Batch upload documents to Azure Search to respect Free tier limits.
    """
    total_docs = len(documents)
    logger.info(f"Uploading {total_docs} documents in batches of {UPLOAD_BATCH_SIZE}...")
    
    for i in range(0, total_docs, UPLOAD_BATCH_SIZE):
        batch = documents[i:i + UPLOAD_BATCH_SIZE]
        logger.info(f"Uploading batch {i//UPLOAD_BATCH_SIZE + 1} ({len(batch)} documents)...")
        try:
            result = client.upload_documents(documents=batch)
            logger.info(f"Successfully uploaded batch {i//UPLOAD_BATCH_SIZE + 1}.")
        except Exception as e:
            logger.error(f"Failed to upload batch {i//UPLOAD_BATCH_SIZE + 1}. Error: {e}")
            
    logger.info("Upload complete.")

def main():
    parser = argparse.ArgumentParser(description="Push local corpus to Azure AI Search.")
    parser.add_argument("--corpus-path", type=str, default="data/corpus.jsonl", help="Path to corpus JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of documents to upload.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generating embeddings (GPU-safe default: 64).")
    args = parser.parse_args()

    # 1. Load configuration and initialize client
    try:
        config = load_azure_config()
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return
        
    client = get_search_client(config['SEARCH_ENDPOINT'], config['SEARCH_API_KEY'], config['SEARCH_INDEX_NAME'])

    # 2. Load corpus
    logger.info(f"Loading corpus from {args.corpus_path}...")
    try:
        corpus_df = load_corpus(args.corpus_path)
    except FileNotFoundError:
        logger.error(f"Corpus file not found at {args.corpus_path}. Please run download_datasets.py first or verify the path.")
        return
        
    if args.limit:
        logger.info(f"Applying limit: Only processing first {args.limit} documents.")
        corpus_df = corpus_df.head(args.limit)

    # 3. Detect Optimal Device and Load Model
    device = get_optimal_device()
    logger.info(f"Loading sentence-transformers/all-MiniLM-L6-v2 on {device.upper()}...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # 4. Prepare documents
    documents = prepare_documents(corpus_df, model, args.batch_size)

    # 5. Upload to Azure
    upload_in_batches(client, documents)

if __name__ == "__main__":
    main()
