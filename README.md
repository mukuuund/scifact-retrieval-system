# Multi-Stage SciFact Retrieval Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://scifact-retrieval-system-ekmitjqr7ze4ke6kpvnvhj.streamlit.app/)

An end-to-end, production-ready system for high-accuracy evidence retrieval within scientific literature. Evaluates and bridges traditional local pipelines with state-of-the-art Azure AI Search infrastructures.

## Problem Statement
Scientific claim verification requires identifying exact sentences within thousands of scientific papers that support or refute a given claim. Standard lexical search engines struggle with this because scientific claims involve dense domain-specific terminology, complex semantics, and subtle negations. This project engineers a multi-stage retrieval architecture capable of maintaining near-perfect Mean Reciprocal Ranks while processing rapidly.

## System Architecture
This repository implements a three-tier retrieval structure:
1. **Local Baselines:** Hand-coded modules mapping the fundamentals of BM25 (Lexical) and FAISS-powered Dense Bi-Encoder vectors.
2. **Cloud Vector Database (Azure AI Search):** Offloads scalable Hybrid searches (RRF Fusion) into Azure infrastructure utilizing dedicated `Hnsw` indexes.
3. **Cognitive Reranking:** Applies a heavy transformer cross-encoder strictly to the Top 10 candidate subset, combining raw efficiency with theoretical accuracy optimization.

## Methods Implemented
- **Local BM25:** `rank_bm25` baseline mechanism.
- **Local Dense:** `SentenceTransformers` (`all-MiniLM-L6-v2`) mapped over a multi-dimensional normalized `FAISS` structure.
- **Local Hybrid:** Reciprocal Rank Fusion of the above mechanisms.
- **Azure AI Search:** Azure's robust semantic models interacting natively via `azure-search-documents`.
- **Cross-Encoder Reranking:** Leveraging `cross-encoder/ms-marco-MiniLM-L-6-v2` for final query-document pair scoring.

## Azure Integration
The system batches and pushes locally-embedded document matrices (via chunked CPU/GPU processing) up to the Azure Cloud explicitly, utilizing an aggressive parallel design pattern designed directly against Free-Tier quota constraints without crashing. 

## Reranking Layer
Dense retrieval acts optimally as a "First-Stage Retriever", pulling broadly matching concepts into a tight funnel. A secondary Cross-Encoder then directly scores the exact nuances in vocabulary syntax. Due to quadratic complexity, this reranker is strategically chained *only* onto the final Top-10 output of the Azure cloud call.

## Evaluation Metrics
- **Recall@K (1, 5, 10):** Evaluates if the *true document* exists anywhere inside the Top $K$ list.
- **MRR@10 (Mean Reciprocal Rank):** Evaluates *how highly* ranked the true document is inside the returned set, penalizing matches that occur lower down the display list.

## Results Table
| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|--------|----------|----------|-----------|--------|
| Local_BM25 | 0.4812 | 0.6120 | 0.6653 | 0.5401 |
| Local_Dense | 0.6540 | 0.8030 | 0.8415 | 0.7100 |
| Local_Hybrid | 0.6811 | 0.8412 | 0.8710 | 0.7392 |
| Azure_Keyword | 0.5120 | 0.6543 | 0.7011 | 0.5693 |
| Azure_Vector | 0.6610 | 0.8080 | 0.8450 | 0.7190 |
| Azure_Hybrid | 0.7198 | **0.9135** | 0.9250 | 0.8293 |
| **Azure_Hybrid_Reranker** | **0.7650** | 0.9287 | **0.9380** | **0.8622** |

## Key Insights
- **Hybrid Fusion dominates pure Dense Search:** Text exact-matches physically ground vector approximations preventing vocabulary drift.
- **Native Cloud Normalization prevails:** Azure's internal scoring standardization outperforms raw local addition of inverse ranks.
- **Precision dictates User Experience:** Reranking minimally impacts top-tier Recall boundaries but powerfully pushes the correct document into the #1 slot, validating its use in high-confidence pipelines.

## Live Demo
**[Launch SciFact Retrieval System 🚀](https://scifact-retrieval-system-ekmitjqr7ze4ke6kpvnvhj.streamlit.app/)**

## Deployment to Streamlit Cloud

This repository is comprehensively configured for Streamlit Community Cloud.

1. Go to [share.streamlit.io](https://share.streamlit.io/) and create a new app.
2. Point it to this GitHub Repository branch and set the **Main file path** to `frontend/app.py`.
3. In the **Advanced Settings > Secrets** block, paste your Azure credentials securely in TOML format:
   ```toml
   AZURE_SEARCH_ENDPOINT = "..."
   AZURE_SEARCH_API_KEY = "..."
   AZURE_SEARCH_INDEX_NAME = "..."
   ```
4. Click **Deploy!**

***

## How to Run

#### 1. Configure the Cloud Database
Ensure your `.env` lists: `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, and `AZURE_SEARCH_INDEX_NAME`.  
```bash
python create_search_index.py
```

#### 2. GPU Batched Embedding Push
Safely bulk upload the SciFact corpus up to Azure (will autodetect CUDA).  
```bash
python push_documents.py --batch-size 64
```

#### 3. General Evaluation Runner
Pings the custom local mechanisms side-by-side against Azure natively. Outputs `final_results.csv`.  
```bash
python compare_azure_vs_local.py
```

#### 4. Advanced System Simulation
Runs the full end-to-end multi-stage pipeline across all valid claims to benchmark MRR@10.  
```bash
python run_reranked_search.py
```

#### 5. Generate Metric Visualizations
```bash
python plot_final_results.py
```

#### 6. Launch the Interactive Frontend Demo (Streamlit)
Spin up the modern local web-interface to test claims naturally and visualize the metrics dashboard.  
```bash
streamlit run frontend/app.py
```

## Future Improvements
- Implement chunking for papers exceeding 600-word abstracts.
- Implement an LLM as a final Generator layer, passing the strictly reranked documents as Context for Retrieval-Augmented Generation (RAG).
