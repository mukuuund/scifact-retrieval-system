import os
import shutil

# 1. Create structure
os.makedirs("src", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2. Move source modules
src_files = ["data_loader.py", "bm25_baseline.py", "dense_retrieval.py", "hybrid_retrieval.py"]
for f in src_files:
    if os.path.exists(f):
        shutil.move(f, f"src/{f}")

if os.path.exists("evaluate.py"):
    shutil.move("evaluate.py", "src/evaluation.py")

open("src/__init__.py", "w").close()

with open("src/utils.py", "w", encoding="utf-8") as f:
    f.write('import os\ndef ensure_dir(p):\n    if not os.path.exists(p): os.makedirs(p)\n')

# 3. Rename old main.py to run_bm25.py
if os.path.exists("main.py"):
    shutil.move("main.py", "run_bm25.py")

# 4. Global string replacement for absolute imports across all runners
for file in ["run_bm25.py", "run_dense.py", "run_hybrid.py", "compare_methods.py", "failure_analysis.py", "plot_results.py"]:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            
        content = content.replace("from data_loader import", "from src.data_loader import")
        content = content.replace("from bm25_baseline import", "from src.bm25_baseline import")
        content = content.replace("from evaluate import", "from src.evaluation import")
        content = content.replace('os.path.join("data",', 'os.path.join(os.path.dirname(__file__), "data",')
        
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)

# 5. Create new main.py with argparse
new_main = """import argparse
import run_bm25
import run_dense
import run_hybrid
import compare_methods
import failure_analysis
import plot_results

def main():
    parser = argparse.ArgumentParser(description="SciFact Retrieval Pipeline Runner")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["bm25", "dense", "hybrid", "compare", "failure", "plot", "all"],
                        help="Execution mode.")
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
        print("Running full pipeline (compare -> failure -> plot)...\\n")
        compare_methods.main()
        print("\\n" + "="*50 + "\\n")
        failure_analysis.main()
        print("\\n" + "="*50 + "\\n")
        plot_results.main()

if __name__ == "__main__":
    main()
"""
with open("main.py", "w", encoding="utf-8") as f:
    f.write(new_main)

# 6. Create README.md
readme = """# SciFact Retrieval Pipeline

A clean, modular, research-quality pipeline for claim verification document retrieval using the SciFact dataset.

## Directory Structure
- `data/` : Raw jsonl dataset files.
- `src/` : Core reusable logic (`data_loader.py`, `bm25_baseline.py`, `dense_retrieval.py`, `hybrid_retrieval.py`, `evaluation.py`, `utils.py`).
- `results/` : Generated evaluation metrics, classification JSONs, and plots.

## Running

The easiest way to run the project is using the central `main.py` entry point:

```bash
# Run individual evaluators
python main.py --mode bm25
python main.py --mode dense
python main.py --mode hybrid

# Run full cross-comparisons
python main.py --mode compare

# View categorical failure buckets
python main.py --mode failure

# Generate visualizations
python main.py --mode plot

# Or run the entire pipeline at once
python main.py --mode all
```
"""
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("Refactoring complete.")
