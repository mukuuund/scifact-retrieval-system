import argparse
import logging
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from sentence_transformers import SentenceTransformer

from src.azure_config import load_azure_config
from src.azure_retrieval import get_search_client, run_hybrid_search, run_keyword_search, run_vector_search

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test search queries against Azure AI Search.")
    parser.add_argument("--query", type=str, default="Machine learning improves cancer diagnosis.", help="Test query string.")
    parser.add_argument("--mode", type=str, choices=["keyword", "vector", "hybrid"], default="hybrid", help="Search mode to execute.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve.")
    args = parser.parse_args()

    # 1. Load config
    try:
        config = load_azure_config()
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return
        
    client = get_search_client(config['SEARCH_ENDPOINT'], config['SEARCH_API_KEY'], config['SEARCH_INDEX_NAME'])

    # 2. Execute selected search mode
    logger.info(f"Executing {args.mode} search against Azure...")
    try:
        if args.mode == "keyword":
            results = run_keyword_search(client, args.query, args.top_k)
        else:
            # Both vector and hybrid require the query to be embedded
            logger.info("Loading local embedding model...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            logger.info(f"Encoding query: '{args.query}'")
            query_vector = model.encode([args.query], normalize_embeddings=True)[0].tolist()
            
            if args.mode == "vector":
                results = run_vector_search(client, query_vector, args.top_k)
            else:
                results = run_hybrid_search(client, args.query, query_vector, args.top_k)
        
        # 3. Print results purely and cleanly
        print(f"\n--- Top {args.top_k} Results for '{args.query}' ({args.mode}) ---")
        if not hasattr(results, "__iter__") or not results:
            print("No results found.")
            return
            
        for idx, result in enumerate(results, start=1):
            print(f"[{idx}] Doc ID: {result['doc_id']} | Score: {result['score']:.4f}")
            print(f"    Title: {result['title']}\n")
            
    except Exception as e:
        logger.error(f"Search failed: {e}")

if __name__ == "__main__":
    main()
