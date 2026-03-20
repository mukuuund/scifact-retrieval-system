import os
import time
import logging
import pandas as pd

# Workaround for sentence-transformers
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from src.data_loader import load_corpus, load_claims
from src.azure_config import load_azure_config
from src.azure_retrieval import get_search_client, run_hybrid_search
from src.evaluation import calc_recall, calc_mrr
from src.reranker import build_reranker, rerank_results
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def evaluate_predictions(preds, ground_truths, method_name, metrics):
    metrics[method_name] = {
        "Recall@5": calc_recall(preds, ground_truths, 5),
        "MRR@10": calc_mrr(preds, ground_truths, 10)  # Evaluating MRR up to max available ranking length
    }

def main():
    logger.info("--- Starting Cross-Encoder Reranking Evaluation ---")
    
    # 1. Setup Data
    corpus_path = os.path.join("data", "corpus.jsonl")
    claims_path = os.path.join("data", "claims_train.jsonl")
    
    if not os.path.exists(corpus_path) or not os.path.exists(claims_path):
        logger.error("Dataset files not found.")
        return

    logger.info("Loading Datasets...")
    corpus_df = load_corpus(corpus_path)
    claims_df = load_claims(claims_path)
    valid_claims = claims_df[claims_df['cited_doc_ids'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    logger.info(f"Loaded {len(corpus_df)} documents and {len(valid_claims)} valid claims.")

    # 2. Setup Client and Embeddings
    try:
        config = load_azure_config()
        azure_client = get_search_client(config['SEARCH_ENDPOINT'], config['SEARCH_API_KEY'], config['SEARCH_INDEX_NAME'])
    except Exception as e:
        logger.error(f"Failed to configure Azure client: {e}")
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Bi-Encoder on {device.upper()}...")
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    # 3. Setup Reranker
    cross_encoder = build_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 4. Storage for evaluation
    ground_truths = []
    azure_hybrid_top5_preds = []
    reranked_top5_preds = []

    logger.info(f"Running Azure Hybrid + Reranking on {len(valid_claims)} claims...")
    
    for _, row in valid_claims.iterrows():
        claim_text = str(row['claim'])
        relevant_ids = [int(i) for i in row['cited_doc_ids']]
        ground_truths.append(relevant_ids)
        
        # We need the vector query
        query_vector = bi_encoder.encode([claim_text], normalize_embeddings=True)[0].tolist()
        
        # Step 1: Initial Azure Hybrid Retrieval (top 10)
        try:
            az_hyb_res = run_hybrid_search(azure_client, claim_text, query_vector, top_k=10)
        except Exception as e:
            logger.warning(f"Azure hybrid search failed for claim: {e}")
            az_hyb_res = []
            
        # Baseline Azure_Hybrid Top 5
        baseline_top5 = az_hyb_res[:5]
        azure_hybrid_top5_preds.append([res['doc_id'] for res in baseline_top5])
        
        # Step 2: Reranking (take top 10 from hybrid, rerank into top 5)
        if az_hyb_res:
            reranked_res = rerank_results(claim_text, az_hyb_res, cross_encoder, corpus_df, top_k=5)
            reranked_top5_preds.append([res['doc_id'] for res in reranked_res])
        else:
            reranked_top5_preds.append([])
            
        time.sleep(0.05) # Rate limit safety

    # 5. Evaluate Performance
    logger.info("Computing metrics...")
    metrics = {}
    
    # Note: MRR is calculated on exactly the predictions provided. If we provide top 5, MRR@10 is essentially MRR@5.
    evaluate_predictions(azure_hybrid_top5_preds, ground_truths, "Azure_Hybrid (Top 5)", metrics)
    evaluate_predictions(reranked_top5_preds, ground_truths, "Azure_Hybrid + Reranker (Top 5)", metrics)

    print(f"\n{'Method':<35} | {'Recall@5':<10} | {'MRR@10':<10}")
    print("-" * 65)
    
    order = ["Azure_Hybrid (Top 5)", "Azure_Hybrid + Reranker (Top 5)"]
    for name in order:
        m = metrics[name]
        print(f"{name:<35} | {m['Recall@5']:<10.4f} | {m['MRR@10']:<10.4f}")

if __name__ == "__main__":
    main()
