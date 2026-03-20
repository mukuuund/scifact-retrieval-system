"""
Module for Hybrid Retrieval using Reciprocal Rank Fusion (RRF).
"""

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from src.bm25_baseline import BM25Retriever
from src.dense_retrieval import retrieve_top_k_dense

def retrieve_top_k_hybrid(query: str, bm25: BM25Retriever, dense_model: SentenceTransformer, dense_index: faiss.Index, corpus_df: pd.DataFrame, k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
    """
    Retrieve top K documents by combining BM25 and Dense retrieval scores using Reciprocal Rank Fusion.
    """
    # Retrieve a larger pool from both methods to compute RRF (deep pool provides better fusion)
    pool_size = max(100, k * 2)
    
    # 1. Get BM25 results
    bm25_doc_ids = bm25.retrieve(query, top_k=pool_size)
    
    # 2. Get Dense results
    dense_results = retrieve_top_k_dense(query, dense_model, dense_index, corpus_df, k=pool_size)
    dense_doc_ids = [res['doc_id'] for res in dense_results]
    
    # 3. Compute RRF scores
    rrf_scores = {}
    
    # Score BM25 ranks
    for rank, doc_id in enumerate(bm25_doc_ids):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
        # Rank is 0-indexed, so we add 1 for the mathematical RRF formula
        rrf_scores[doc_id] += 1.0 / ((rank + 1) + rrf_k)
        
    # Score Dense ranks
    for rank, doc_id in enumerate(dense_doc_ids):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += 1.0 / ((rank + 1) + rrf_k)
        
    # 4. Sort documents by RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 5. Get top K
    top_k_docs = sorted_docs[:k]
    
    # 6. Format results identically to other methods
    results = []
    for doc_id, score in top_k_docs:
        # Simple lookup strategy (optimized for beginner readability)
        doc_row = corpus_df[corpus_df['doc_id'] == doc_id].iloc[0]
        results.append({
            "doc_id": doc_id,
            "title": doc_row.get("title", ""),
            "score": score
        })
        
    return results

def recall_at_k_hybrid(claims_df: pd.DataFrame, corpus_df: pd.DataFrame, bm25: BM25Retriever, dense_model: SentenceTransformer, dense_index: faiss.Index, k: int = 5) -> float:
    """Evaluate Recall@K for the hybrid retriever."""
    recalls = []
    
    for _, row in claims_df.iterrows():
        claim_text = row.get("claim", "")
        cited_ids = row.get("cited_doc_ids", [])
        
        if not cited_ids or pd.isna(claim_text) or not str(claim_text).strip():
            continue
            
        top_results = retrieve_top_k_hybrid(claim_text, bm25, dense_model, dense_index, corpus_df, k=k)
        retrieved_ids = [res["doc_id"] for res in top_results]
        
        hits = set(retrieved_ids).intersection(set(cited_ids))
        recall = len(hits) / len(cited_ids)
        recalls.append(recall)
        
    return sum(recalls) / len(recalls) if recalls else 0.0

def mean_reciprocal_rank_hybrid(claims_df: pd.DataFrame, corpus_df: pd.DataFrame, bm25: BM25Retriever, dense_model: SentenceTransformer, dense_index: faiss.Index, k: int = 10) -> float:
    """Evaluate MRR@K for the hybrid retriever."""
    reciprocal_ranks = []
    
    for _, row in claims_df.iterrows():
        claim_text = row.get("claim", "")
        cited_ids = row.get("cited_doc_ids", [])
        
        if not cited_ids or pd.isna(claim_text) or not str(claim_text).strip():
            continue
            
        top_results = retrieve_top_k_hybrid(claim_text, bm25, dense_model, dense_index, corpus_df, k=k)
        retrieved_ids = [res["doc_id"] for res in top_results]
        cited_set = set(cited_ids)
        
        rank = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in cited_set:
                rank = i + 1
                break
                
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
