"""
Runner for BM25 baseline.
"""
import os
from src.data_loader import load_corpus, load_claims
from src.bm25_baseline import BM25Retriever
from src.evaluation import calc_recall, calc_mrr

def main():
    corpus_df = load_corpus(os.path.join(os.path.dirname(__file__), "data", "corpus.jsonl"))
    claims_df = load_claims(os.path.join(os.path.dirname(__file__), "data", "claims_train.jsonl"))
    valid_claims = claims_df[claims_df['cited_doc_ids'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    
    bm25 = BM25Retriever(corpus_df)
    preds = []
    gts = []
    
    print("--- Running BM25 Retrieval Analysis ---")
    for _, row in valid_claims.iterrows():
        gts.append([int(x) for x in row['cited_doc_ids']])
        b_top = bm25.retrieve(row['claim'], top_k=10)
        preds.append([int(i) for i in b_top])
        
    print("--- BM25 Evaluation ---")
    print(f"Recall@1:  {calc_recall(preds, gts, 1):.4f}")
    print(f"Recall@5:  {calc_recall(preds, gts, 5):.4f}")
    print(f"Recall@10: {calc_recall(preds, gts, 10):.4f}")
    print(f"MRR@10:    {calc_mrr(preds, gts, 10):.4f}")

if __name__ == "__main__":
    main()
