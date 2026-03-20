import os
import json
import logging
import pandas as pd
import time

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from src.data_loader import load_corpus, load_claims
from src.bm25_baseline import BM25Retriever
from src.dense_retrieval import build_dense_index, retrieve_top_k_dense
from src.hybrid_retrieval import retrieve_top_k_hybrid
from src.evaluation import calc_recall, calc_mrr

from sentence_transformers import SentenceTransformer
from src.azure_config import load_azure_config
from src.azure_retrieval import get_search_client, run_keyword_search, run_vector_search, run_hybrid_search

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def evaluate_predictions(preds, ground_truths, method_name, metrics):
    metrics[method_name] = {
        "Recall@1": calc_recall(preds, ground_truths, 1),
        "Recall@5": calc_recall(preds, ground_truths, 5),
        "Recall@10": calc_recall(preds, ground_truths, 10),
        "MRR@10": calc_mrr(preds, ground_truths, 10)
    }

def main():
    logger.info("--- Starting Comparison ---")
    
    # 1. Setup local data
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

    # 2. Setup Local Retrievers
    logger.info("Initializing Local Retrievers...")
    bm25 = BM25Retriever(corpus_df)
    model, index = build_dense_index(corpus_df, text_column="text", model_name="all-MiniLM-L6-v2")

    # 3. Setup Azure Client
    logger.info("Initializing Azure Search Client...")
    try:
        config = load_azure_config()
        azure_client = get_search_client(config['SEARCH_ENDPOINT'], config['SEARCH_API_KEY'], config['SEARCH_INDEX_NAME'])
    except Exception as e:
        logger.error(f"Failed to configure Azure client: {e}")
        return

    # Predictions storage
    ground_truths = []
    
    local_bm25_preds = []
    local_dense_preds = []
    local_hybrid_preds = []
    
    azure_keyword_preds = []
    azure_vector_preds = []
    azure_hybrid_preds = []

    logger.info(f"Running Retrieval on {len(valid_claims)} claims...")
    for idx, row in valid_claims.iterrows():
        claim_text = str(row['claim'])
        relevant_ids = [int(i) for i in row['cited_doc_ids']]
        ground_truths.append(relevant_ids)
        
        # --- Local Retrievals ---
        # BM25
        local_bm25_top = [int(i) for i in bm25.retrieve(claim_text, top_k=10)]
        local_bm25_preds.append(local_bm25_top)
        
        # Dense
        dense_results = retrieve_top_k_dense(claim_text, model, index, corpus_df, k=10)
        local_dense_top = [int(res['doc_id']) for res in dense_results]
        local_dense_preds.append(local_dense_top)
        
        # Hybrid
        hybrid_results = retrieve_top_k_hybrid(claim_text, bm25, model, index, corpus_df, k=10, rrf_k=60)
        local_hybrid_top = [int(res['doc_id']) for res in hybrid_results]
        local_hybrid_preds.append(local_hybrid_top)
        
        # --- Azure Retrievals ---
        # Azure Keyword
        try:
            az_kwd_res = run_keyword_search(azure_client, claim_text, top_k=10)
            azure_keyword_preds.append([int(r['doc_id']) for r in az_kwd_res])
        except Exception as e:
            logger.warning(f"Azure keyword search failed for claim: {e}")
            azure_keyword_preds.append([])
            
        # Azure Vector & Hybrid requires embedded vector
        query_vector = model.encode([claim_text], normalize_embeddings=True)[0].tolist()
        
        try:
            az_vec_res = run_vector_search(azure_client, query_vector, top_k=10)
            azure_vector_preds.append([int(r['doc_id']) for r in az_vec_res])
        except Exception as e:
            logger.warning(f"Azure vector search failed for claim: {e}")
            azure_vector_preds.append([])
            
        try:
            az_hyb_res = run_hybrid_search(azure_client, claim_text, query_vector, top_k=10)
            azure_hybrid_preds.append([int(r['doc_id']) for r in az_hyb_res])
        except Exception as e:
            logger.warning(f"Azure hybrid search failed for claim: {e}")
            azure_hybrid_preds.append([])
            
        # Sleep slightly to avoid spamming the free tier heavily
        time.sleep(0.1)

    logger.info("Computing metrics...")
    metrics = {}
    evaluate_predictions(local_bm25_preds, ground_truths, "Local_BM25", metrics)
    evaluate_predictions(local_dense_preds, ground_truths, "Local_Dense", metrics)
    evaluate_predictions(local_hybrid_preds, ground_truths, "Local_Hybrid", metrics)
    evaluate_predictions(azure_keyword_preds, ground_truths, "Azure_Keyword", metrics)
    evaluate_predictions(azure_vector_preds, ground_truths, "Azure_Vector", metrics)
    evaluate_predictions(azure_hybrid_preds, ground_truths, "Azure_Hybrid", metrics)

    print(f"\n{'Method':<15} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR@10':<10}")
    print("-" * 70)
    
    order = ["Local_BM25", "Azure_Keyword", "Local_Dense", "Azure_Vector", "Local_Hybrid", "Azure_Hybrid"]
    
    for name in order:
        m = metrics[name]
        print(f"{name:<15} | {m['Recall@1']:<10.4f} | {m['Recall@5']:<10.4f} | {m['Recall@10']:<10.4f} | {m['MRR@10']:<10.4f}")

    # 4. Save to CSV
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    records = []
    for name in order:
        records.append({
            "Method": name,
            "Recall@1": round(metrics[name]["Recall@1"], 4),
            "Recall@5": round(metrics[name]["Recall@5"], 4),
            "Recall@10": round(metrics[name]["Recall@10"], 4),
            "MRR@10": round(metrics[name]["MRR@10"], 4)
        })
        
    out_path = os.path.join(results_dir, "azure_vs_local_comparison.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    logger.info(f"Results successfully saved to {out_path}.")

if __name__ == "__main__":
    main()
