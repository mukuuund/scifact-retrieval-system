Viewed README.md:1-86

It looks like you might have accidentally copied the *rendered* version of the README from GitHub (which strips away the underlying formatting like bolding, tables, and headers)!

You do not need to rewrite or fix anything. Because of the code changes we made step previously, **your local [README.md](cci:7://file:///c:/Users/Mukun/OneDrive/Desktop/Coding/Azure%20Project/README.md:0:0-0:0) file actually still has all the perfect formatting already programmed in.** 

If you'd like to copy it, here is the exact full, complete markdown of your [README.md](cci:7://file:///c:/Users/Mukun/OneDrive/Desktop/Coding/Azure%20Project/README.md:0:0-0:0) file as it currently exists on your computer:

```markdown
# Multi-Stage SciFact Retrieval Pipeline

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

## Deployment to Azure App Service (Linux)

This repository is ready to be deployed to Azure App Service via GitHub.

1. Push this repository to GitHub.
2. In the Azure Portal, create a new **Web App** (Publish: Code, Runtime stack: Python 3.1x, OS: Linux).
3. In the Web App's **Deployment Center**, connect your GitHub repository to enable CI/CD.
4. Go to **Settings > Environment variables** (or Configuration) and add the following App settings:
   - `AZURE_SEARCH_ENDPOINT`
   - `AZURE_SEARCH_API_KEY`
   - `AZURE_SEARCH_INDEX_NAME`
   - `AZURE_STORAGE_CONNECTION_STRING` (If needed)
5. Set the Startup Command (under Configuration -> General Settings) to: `bash run.sh`
6. Azure App Service will install dependencies from [requirements.txt](cci:7://file:///c:/Users/Mukun/OneDrive/Desktop/Coding/Azure%20Project/requirements.txt:0:0-0:0) and launch the Streamlit frontend.

***

## How to Run
#### 1. Configure the Cloud Database
Ensure your [.env](cci:7://file:///c:/Users/Mukun/OneDrive/Desktop/Coding/Azure%20Project/.env:0:0-0:0) lists: `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, and `AZURE_SEARCH_INDEX_NAME`.  
`python create_search_index.py`
#### 2. GPU Batched Embedding Push
Safely bulk upload the SciFact corpus up to Azure (will autodetect CUDA).  
`python push_documents.py --batch-size 64`
#### 3. General Evaluation Runner
Pings the custom local mechanisms side-by-side against Azure natively. Outputs `final_results.csv`.  
`python compare_azure_vs_local.py`
#### 4. Advanced System Simulation
Runs the full end-to-end multi-stage pipeline across all valid claims to benchmark MRR@10.  
`python run_reranked_search.py`
#### 5. Generate Metric Visualizations
`python plot_final_results.py`
#### 6. Launch the Interactive Frontend Demo (Streamlit)
Spin up the modern local web-interface to test claims naturally and visualize the metrics dashboard.  
`streamlit run frontend/app.py`

## Future Improvements
- Implement chunking for papers exceeding 600-word abstracts.
- Implement an LLM as a final Generator layer, passing the strictly reranked documents as Context for Retrieval-Augmented Generation (RAG).
```
