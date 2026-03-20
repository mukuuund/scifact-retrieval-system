import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def build_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """
    Initialize the Cross-Encoder model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CrossEncoder: {model_name} on {device.upper()}...")
    return CrossEncoder(model_name, device=device)

def rerank_results(query: str, results: List[Dict[str, Any]], model: CrossEncoder, corpus_df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Rerank a list of initial search results using a cross-encoder model.
    
    Args:
        query (str): The search query.
        results (List[Dict[str, Any]]): Top-N results from the initial retriever.
        model (CrossEncoder): The initialized cross-encoder model.
        corpus_df (pd.DataFrame): The corpus DataFrame to look up full texts.
        top_k (int): The number of final top results to return.
        
    Returns:
        List[Dict[str, Any]]: The re-ordered list of top_k results.
    """
    if not results:
        return []
        
    # Build query-document pairs
    cross_inp = []
    
    # Fast filtering for corpus texts
    valid_doc_ids = {res["doc_id"] for res in results}
    sub_corpus = corpus_df[corpus_df['doc_id'].isin(valid_doc_ids)]
    text_mapping = dict(zip(sub_corpus['doc_id'], sub_corpus['text']))
    
    for res in results:
        doc_id = res['doc_id']
        doc_text = text_mapping.get(doc_id, res.get('title', ''))
        cross_inp.append((query, doc_text))
        
    # Score pairs
    scores = model.predict(cross_inp)
    
    # Assign scores and sort
    for idx, res in enumerate(results):
        res["cross_score"] = float(scores[idx])
        
    reranked = sorted(results, key=lambda x: x["cross_score"], reverse=True)
    return reranked[:top_k]
