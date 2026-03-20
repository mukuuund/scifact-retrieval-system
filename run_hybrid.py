"""
Runner for Hybrid Retrieval baseline.
"""
import os
from src.data_loader import load_corpus, load_claims
from src.bm25_baseline import BM25Retriever
from src.dense_retrieval import build_dense_index
from src.hybrid_retrieval import retrieve_top_k_hybrid
from src.evaluation import calc_recall, calc_mrr

def main():
    corpus_df = load_corpus(os.path.join(os.path.dirname(__file__), "data", "corpus.jsonl"))
    claims_df = load_claims(os.path.join(os.path.dirname(__file__), "data", "claims_train.jsonl"))
    valid_claims = claims_df[claims_df['cited_doc_ids'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    
    print("--- Initializing Retrievers ---")
    bm25 = BM25Retriever(corpus_df)
    model, index = build_dense_index(corpus_df, text_column="text", model_name="all-MiniLM-L6-v2")
    preds = []
    gts = []
    
    print("--- Running Hybrid Retrieval Analysis ---")
    for _, row in valid_claims.iterrows():
        gts.append([int(x) for x in row['cited_doc_ids']])
        h_res = retrieve_top_k_hybrid(row['claim'], bm25, model, index, corpus_df, k=10)
        preds.append([int(r['doc_id']) for r in h_res])
        
    print("--- Hybrid Evaluation ---")
    print(f"Recall@1:  {calc_recall(preds, gts, 1):.4f}")
    print(f"Recall@5:  {calc_recall(preds, gts, 5):.4f}")
    print(f"Recall@10: {calc_recall(preds, gts, 10):.4f}")
    print(f"MRR@10:    {calc_mrr(preds, gts, 10):.4f}")

if __name__ == "__main__":
    main()
