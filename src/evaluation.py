"""
Centralized evaluation metrics for retrieval.
"""
from typing import List

def calc_recall(predictions: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    """Calculate the average Recall@K."""
    if not predictions or not ground_truth:
        return 0.0
    recalls = []
    for retrieved_ids, relevant_ids in zip(predictions, ground_truth):
        top_k = retrieved_ids[:k]
        hits = set(top_k).intersection(set(relevant_ids))
        if relevant_ids:
            recalls.append(len(hits) / len(relevant_ids))
    return sum(recalls) / len(recalls) if recalls else 0.0

def calc_mrr(predictions: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    """Calculate the Mean Reciprocal Rank@K."""
    if not predictions or not ground_truth:
        return 0.0
    mrrs = []
    for retrieved_ids, relevant_ids in zip(predictions, ground_truth):
        top_k = retrieved_ids[:k]
        rank = 0
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(top_k):
            if doc_id in relevant_set:
                rank = i + 1
                break
        mrrs.append(1.0 / rank if rank > 0 else 0.0)
    return sum(mrrs) / len(mrrs) if mrrs else 0.0
