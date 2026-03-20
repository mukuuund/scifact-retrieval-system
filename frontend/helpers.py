import os
import streamlit as st
import torch
from typing import List, Dict, Any, Tuple

# Fix transformers / tensorflow warning
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
# Fix OpenMP conflict between Faiss and PyTorch on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.data_loader import load_corpus
from src.bm25_baseline import BM25Retriever
from src.dense_retrieval import build_dense_index, retrieve_top_k_dense
from src.hybrid_retrieval import retrieve_top_k_hybrid
from src.azure_config import load_azure_config
from src.azure_retrieval import get_search_client, run_keyword_search, run_vector_search, run_hybrid_search
from src.reranker import build_reranker, rerank_results
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------
# CACHED MODEL LOADERS
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading SciFact Corpus...")
def load_cached_corpus():
    corpus_path = os.path.join("data", "corpus.jsonl")
    if not os.path.exists(corpus_path):
        st.error(f"Missing {corpus_path}. Please run download_datasets.py first.")
        st.stop()
    return load_corpus(corpus_path)

@st.cache_resource(show_spinner="Loading Local Retrievers (BM25 + FAISS)...")
def load_local_retrievers(_corpus_df):
    bm25 = BM25Retriever(_corpus_df)
    model, index = build_dense_index(_corpus_df, text_column="text", model_name="all-MiniLM-L6-v2")
    return bm25, model, index

@st.cache_resource(show_spinner="Loading Cross-Encoder Reranker...")
def load_cached_reranker():
    return build_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource(show_spinner="Connecting to Azure AI Search...")
def load_cached_azure_client():
    try:
        config = load_azure_config()
        client = get_search_client(config['SEARCH_ENDPOINT'], config['SEARCH_API_KEY'], config['SEARCH_INDEX_NAME'])
        return client
    except Exception as e:
        st.warning(f"⚠️ **Azure Connection Failed:** {e}\n\nAzure AI Search algorithms will be disabled in the UI.")
        st.info("💡 **Running on Streamlit Cloud?**\nGo to your deployed app's **Settings > Advanced > Secrets** and paste your Azure variables:\n```toml\nAZURE_SEARCH_ENDPOINT=\"...\"\nAZURE_SEARCH_API_KEY=\"...\"\nAZURE_SEARCH_INDEX_NAME=\"...\"\nAZURE_STORAGE_CONNECTION_STRING=\"...\"\n```")
        return None

# -------------------------------------------------------------
# ALGORITHM ROUTER
# -------------------------------------------------------------
def execute_search(
    query: str, 
    method: str, 
    top_k: int, 
    corpus_df, 
    bm25, 
    dense_model, 
    faiss_index, 
    azure_client, 
    cross_encoder
) -> List[Dict[str, Any]]:
    """Maps the frontend select_box choice to the actual backend pipeline execution."""
    
    if not query.strip():
        return []

    # Map IDs to actual titles/abstracts for display regardless of retriever
    def enrich_results(res_list):
        enriched = []
        for r in res_list:
            doc_id = r['doc_id']
            # Find in corpus
            match = corpus_df[corpus_df['doc_id'] == doc_id]
            if not match.empty:
                row = match.iloc[0]
                enriched.append({
                    'doc_id': doc_id,
                    'title': row.get('title', r.get('title', 'Unknown Title')),
                    'abstract': row.get('abstract_text', 'No abstract available.'),
                    'score': r.get('score', 0.0),
                    'method': method
                })
            else:
                enriched.append(r)
        return enriched

    try:
        # 1. Local Methods
        if method == "Local_BM25":
            ids = bm25.retrieve(query, top_k=top_k)
            # Formatting to match standard dict
            raw_res = [{'doc_id': int(i), 'score': 0.0} for i in ids]
            return enrich_results(raw_res)
            
        elif method == "Local_Dense":
            raw_res = retrieve_top_k_dense(query, dense_model, faiss_index, corpus_df, k=top_k)
            return enrich_results(raw_res)
            
        elif method == "Local_Hybrid":
            raw_res = retrieve_top_k_hybrid(query, bm25, dense_model, faiss_index, corpus_df, k=top_k)
            return enrich_results(raw_res)
            
        # 2. Azure Methods
        elif method.startswith("Azure"):
            if azure_client is None:
                raise ValueError("Azure client is not connected.")
                
            if method == "Azure_Keyword":
                raw_res = run_keyword_search(azure_client, query, top_k=top_k)
                return enrich_results(raw_res)
                
            # Both vector and hybrid need an embedded query
            query_vector = dense_model.encode([query], normalize_embeddings=True)[0].tolist()
            
            if method == "Azure_Vector":
                raw_res = run_vector_search(azure_client, query_vector, top_k=top_k)
                return enrich_results(raw_res)
                
            elif method == "Azure_Hybrid":
                raw_res = run_hybrid_search(azure_client, query, query_vector, top_k=top_k)
                return enrich_results(raw_res)
                
            elif method == "Azure_Hybrid_Reranker":
                # Step 1: Broad search Top 10
                broad_res = run_hybrid_search(azure_client, query, query_vector, top_k=10)
                # Step 2: Rerank down to top_k
                if not broad_res:
                    return []
                reranked_res = rerank_results(query, broad_res, cross_encoder, corpus_df, top_k=top_k)
                # Ensure the scores mapped back reflect the cross_score in the dict
                for r in reranked_res:
                    r['score'] = r.get('cross_score', r.get('score', 0.0))
                return enrich_results(reranked_res)

        else:
            return []
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise Exception(f"Search failed for {method}: {str(e)}")
