import argparse
import run_bm25
import run_dense
import run_hybrid
import compare_methods
import failure_analysis
import plot_results

def main():
    parser = argparse.ArgumentParser(description="SciFact Retrieval Pipeline Runner")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["bm25", "dense", "hybrid", "compare", "failure", "plot", "all"],
                        help="Execution mode (defaults to 'all').")
    args = parser.parse_args()
    
    if args.mode == "bm25":
        run_bm25.main()
    elif args.mode == "dense":
        run_dense.main()
    elif args.mode == "hybrid":
        run_hybrid.main()
    elif args.mode == "compare":
        compare_methods.main()
    elif args.mode == "failure":
        failure_analysis.main()
    elif args.mode == "plot":
        plot_results.main()
    elif args.mode == "all":
        print("Running full pipeline (compare -> failure -> plot)...\n")
        compare_methods.main()
        print("\n" + "="*50 + "\n")
        failure_analysis.main()
        print("\n" + "="*50 + "\n")
        plot_results.main()

if __name__ == "__main__":
    main()
