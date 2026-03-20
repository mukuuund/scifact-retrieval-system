"""
Analyze and categorize retrieval failures across different methods efficiently.
"""

import os
import json

def main():
    preds_path = os.path.join("results", "predictions.json")
    if not os.path.exists(preds_path):
        print("Error: Please run 'python main.py --mode compare' first to generate predictions.")
        return
        
    with open(preds_path, "r", encoding="utf-8") as f:
        raw_predictions = json.load(f)
        
    categories = {
        "bm25_failed_dense_succeeded": [],
        "dense_failed_bm25_succeeded": [],
        "hybrid_succeeded_both_failed": [],
        "all_methods_failed": []
    }
    
    for record in raw_predictions:
        gold = set(record["gold_doc_ids"])
        b_succ = len(set(record["bm25_top10"][:5]).intersection(gold)) > 0
        d_succ = len(set(record["dense_top10"][:5]).intersection(gold)) > 0
        h_succ = len(set(record["hybrid_top10"][:5]).intersection(gold)) > 0
        
        rec = {
            "claim": record["claim"],
            "gold_doc_ids": record["gold_doc_ids"],
            "bm25_top5": record["bm25_top10"][:5],
            "dense_top5": record["dense_top10"][:5],
            "hybrid_top5": record["hybrid_top10"][:5]
        }
        
        if not b_succ and d_succ: categories["bm25_failed_dense_succeeded"].append(rec)
        if not d_succ and b_succ: categories["dense_failed_bm25_succeeded"].append(rec)
        if h_succ and not b_succ and not d_succ: categories["hybrid_succeeded_both_failed"].append(rec)
        if not b_succ and not d_succ and not h_succ: categories["all_methods_failed"].append(rec)
            
    out_path = os.path.join("results", "failure_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=4)
        
    print(f"Failure analysis correctly mapped and extracted.")
    print(f"File stored safely to '{out_path}'.")
    
    print("\nTotal Counts:")
    for cat, items in categories.items():
        print(f" - {cat}: {len(items)}")

if __name__ == "__main__":
    main()
