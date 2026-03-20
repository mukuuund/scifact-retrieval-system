"""
Compare BM25, Dense, and Hybrid Retrieval on SciFact valid claims.
"""

import os
import json
import pandas as pd
from src.data_loader import load_corpus, load_claims
from src.bm25_baseline import BM25Retriever
from src.dense_retrieval import build_dense_index, retrieve_top_k_dense
from src.hybrid_retrieval import retrieve_top_k_hybrid
from src.evaluation import calc_recall, calc_mrr

def main():
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "corpus.jsonl")
    claims_path = os.path.join(os.path.dirname(__file__), "data", "claims_train.jsonl")
    
    if not os.path.exists(corpus_path) or not os.path.exists(claims_path):
        print("Dataset files not found.")
        return

    print("--- Loading Datasets ---")
    corpus_df = load_corpus(corpus_path)
    claims_df = load_claims(claims_path)
    
    valid_claims = claims_df[claims_df['cited_doc_ids'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    print(f"Loaded {len(corpus_df)} documents and {len(valid_claims)} valid claims.")

    print("\n--- Initializing Retrievers ---")
    bm25 = BM25Retriever(corpus_df)
    model, index = build_dense_index(corpus_df, text_column="text", model_name="all-MiniLM-L6-v2")

    print("\n--- Running Retrieval for all Valid Claims ---")
    bm25_preds, dense_preds, hybrid_preds, ground_truths = [], [], [], []
    raw_predictions = []
    
    for _, row in valid_claims.iterrows():
        claim_text = str(row['claim'])
        relevant_ids = [int(i) for i in row['cited_doc_ids']]
        ground_truths.append(relevant_ids)
        
        # We enforce casting to standard python integer handles here to prevent numpy/FAISS int32 serialization bugs later
        bm25_top100 = bm25.retrieve(claim_text, top_k=100)
        bm25_top10 = [int(i) for i in bm25_top100[:10]]
        bm25_preds.append(bm25_top10)
        
        dense_results = retrieve_top_k_dense(claim_text, model, index, corpus_df, k=100)
        dense_top10 = [int(res['doc_id']) for res in dense_results[:10]]
        dense_preds.append(dense_top10)
        
        hybrid_results = retrieve_top_k_hybrid(claim_text, bm25, model, index, corpus_df, k=10, rrf_k=60)
        hybrid_top10 = [int(res['doc_id']) for res in hybrid_results]
        hybrid_preds.append(hybrid_top10)
        
        raw_predictions.append({
            "claim": claim_text,
            "gold_doc_ids": relevant_ids,
            "bm25_top10": bm25_top10,
            "dense_top10": dense_top10,
            "hybrid_top10": hybrid_top10
        })
        
    print("\n--- Evaluation Results ---")
    metrics = {}
    for name, preds in [("BM25", bm25_preds), ("Dense", dense_preds), ("Hybrid", hybrid_preds)]:
        metrics[name] = {
            "R@1": calc_recall(preds, ground_truths, 1),
            "R@5": calc_recall(preds, ground_truths, 5),
            "R@10": calc_recall(preds, ground_truths, 10),
            "MRR@10": calc_mrr(preds, ground_truths, 10)
        }
    
    print(f"\n{'Method':<10} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR@10':<10}")
    print("-" * 65)
    for name in ["BM25", "Dense", "Hybrid"]:
        m = metrics[name]
        print(f"{name:<10} | {m['R@1']:<10.4f} | {m['R@5']:<10.4f} | {m['R@10']:<10.4f} | {m['MRR@10']:<10.4f}")

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    records = []
    for name in ["BM25", "Dense", "Hybrid"]:
        m = metrics[name]
        records.append({
            "Method": name,
            "Recall@1": round(m["R@1"], 4),
            "Recall@5": round(m["R@5"], 4),
            "Recall@10": round(m["R@10"], 4),
            "MRR@10": round(m["MRR@10"], 4)
        })
    pd.DataFrame(records).to_csv(os.path.join(results_dir, "metrics_comparison.csv"), index=False)
    
    with open(os.path.join(results_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(raw_predictions, f, indent=4)
        
    print("\nResults successfully saved to 'results/metrics_comparison.csv' and 'results/predictions.json'.")

if __name__ == "__main__":
    main()
