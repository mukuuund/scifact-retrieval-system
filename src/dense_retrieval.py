"""
Module for Dense Retrieval using Sentence Transformers and FAISS.
"""

import os
# Force transformers to ignore any broken tensorflow installation
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any

def build_dense_index(corpus_df: pd.DataFrame, text_column: str = "text", model_name: str = "all-MiniLM-L6-v2") -> Tuple[SentenceTransformer, faiss.Index]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    texts = corpus_df[text_column].fillna("").tolist()
    
    # Encode with normalization. Turned off progress bar for clean logs.
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return model, index

def retrieve_top_k_dense(query: str, model: SentenceTransformer, index: faiss.Index, corpus_df: pd.DataFrame, k: int = 5) -> List[Dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        return []

    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_vector).astype("float32")
    
    scores, indices = index.search(query_vector, k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(corpus_df):
            doc_row = corpus_df.iloc[idx]
            result = {
                "doc_id": doc_row["doc_id"],
                "title": doc_row.get("title", ""),
                "score": float(score)
            }
            results.append(result)
            
    return results

def recall_at_k_dense(claims_df: pd.DataFrame, corpus_df: pd.DataFrame, model: SentenceTransformer, index: faiss.Index, k: int = 5) -> float:
    recalls = []
    
    for _, row in claims_df.iterrows():
        claim_text = row.get("claim", "")
        cited_ids = row.get("cited_doc_ids", [])
        
        if not cited_ids or pd.isna(claim_text) or not str(claim_text).strip():
            continue
            
        top_results = retrieve_top_k_dense(claim_text, model, index, corpus_df, k=k)
        retrieved_ids = [res["doc_id"] for res in top_results]
        
        hits = set(retrieved_ids).intersection(set(cited_ids))
        recall = len(hits) / len(cited_ids)
        recalls.append(recall)
        
    if not recalls:
        return 0.0
        
    return sum(recalls) / len(recalls)

def mean_reciprocal_rank_dense(claims_df: pd.DataFrame, corpus_df: pd.DataFrame, model: SentenceTransformer, index: faiss.Index, k: int = 10) -> float:
    reciprocal_ranks = []
    
    for _, row in claims_df.iterrows():
        claim_text = row.get("claim", "")
        cited_ids = row.get("cited_doc_ids", [])
        
        if not cited_ids or pd.isna(claim_text) or not str(claim_text).strip():
            continue
            
        top_results = retrieve_top_k_dense(claim_text, model, index, corpus_df, k=k)
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
            
    if not reciprocal_ranks:
        return 0.0
        
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
