import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    results_path = os.path.join("results", "final_results.csv")
    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return

    # Load data
    df = pd.read_csv(results_path)

    # Set up styling
    sns.set_theme(style="whitegrid")
    
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Plot Recall@5
    sns.barplot(data=df, x="Recall@5", y="Method", ax=axes[0], palette="Blues_d")
    axes[0].set_title("Recall@5 Comparison")
    axes[0].set_xlim(0, 1.0)
    for i, v in enumerate(df["Recall@5"]):
        axes[0].text(v + 0.02, i, str(v), color='black', va='center', fontweight='bold')

    # 2. Plot MRR@10
    sns.barplot(data=df, x="MRR@10", y="Method", ax=axes[1], palette="Greens_d")
    axes[1].set_title("MRR@10 Comparison")
    axes[1].set_xlim(0, 1.0)
    # Hide the y-axis labels on the right plot to avoid duplicate text overlapping
    axes[1].set_ylabel("")
    axes[1].set_yticks([]) 
    for i, v in enumerate(df["MRR@10"]):
        axes[1].text(v + 0.02, i, str(v), color='black', va='center', fontweight='bold')

    # Prevent layout overlapping
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join("results", "final_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to {output_path}")

if __name__ == "__main__":
    main()
