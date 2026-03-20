"""
Plot the retrieval metrics from results/metrics_comparison.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    csv_path = os.path.join("results", "metrics_comparison.csv")
    out_path = os.path.join("results", "retrieval_comparison.png")
    
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found.")
        print("Please wait for compare_methods.py to finish saving the metrics.")
        return

    # Load data
    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    # We expect columns: Method, Recall@1, Recall@5, Recall@10, MRR@10
    metrics = ["Recall@1", "Recall@5", "Recall@10", "MRR@10"]
    methods = df["Method"].tolist()
    
    # Setup plot parameters
    x = np.arange(len(metrics))  # The label locations on X axis
    width = 0.25                 # The width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))

    # Modern, distinct colors for BM25, Dense, and Hybrid
    colors = ['#4A90E2', '#50E3C2', '#F5A623'] 
    
    # Plot grouped bars
    for i, method in enumerate(methods):
        # Grab the values for this specific method
        row = df[df["Method"] == method]
        values = [row[m].values[0] for m in metrics]
        
        # Calculate dynamic offset so bars group closely around each tick mark
        offset = (i - len(methods)/2 + 0.5) * width
        
        # Plot the bar
        rects = ax.bar(x + offset, values, width, label=method, color=colors[i % len(colors)])
        
        # Attach a text label above each bar
        ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=9)

    # Styling and Labels
    ax.set_ylabel('Scores')
    ax.set_title('SciFact Retrieval Baseline Evaluation', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)  # Scale up slightly strictly to make room for text labels
    
    # Place legend cleanly outside the plot area
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(methods))

    # Adjust layout to prevent text clipping
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to '{out_path}'.")

if __name__ == "__main__":
    main()
