import os
import streamlit as st
import pandas as pd
from PIL import Image

# Use absolute imports dynamically by modifying sys path if needed (though running from root solves this)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.helpers import (
    load_cached_corpus, 
    load_local_retrievers, 
    load_cached_reranker, 
    load_cached_azure_client,
    execute_search
)

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="SciFact Retrieval Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# BACKGROUND CACHING & SETUP
# -------------------------------------------------------------
corpus_df = load_cached_corpus()
bm25, dense_model, faiss_index = load_local_retrievers(corpus_df)
cross_encoder = load_cached_reranker()
azure_client = load_cached_azure_client()

ALL_METHODS = [
    "Local_BM25",
    "Local_Dense",
    "Local_Hybrid",
    "Azure_Keyword",
    "Azure_Vector",
    "Azure_Hybrid",
    "Azure_Hybrid_Reranker"
]

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.title("🔬 SciFact Retrieval System")
st.markdown("""
Welcome to the multi-stage scientific claim verification engine. 
This dashboard allows you to execute end-to-end evidence retrieval against the SciFact corpus, comparing purely local lexical/dense baseline algorithms against a cloud-native **Azure AI Search** hybrid vector configuration, capped off with a heavy Transformer **Cross-Encoder Reranking** layer.
""")
st.divider()

# -------------------------------------------------------------
# TABS
# -------------------------------------------------------------
tab_search, tab_compare, tab_metrics = st.tabs(["🔍 Interactive Search", "⚖️ Compare Methods", "📊 Metrics Dashboard"])

# ==========================================
# TAB 1: INTERACTIVE SEARCH
# ==========================================
with tab_search:
    st.subheader("Query the Database")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area("Enter a scientific claim:", "Machine learning improves cancer diagnosis.", height=100)
    with col2:
        method = st.selectbox("Retrieval Engine:", ALL_METHODS, index=6) # Default to best
        top_k = st.slider("Top K Results:", min_value=1, max_value=20, value=5)
        search_button = st.button("🚀 Run Search")

    if search_button:
        if not query.strip():
            st.error("Please enter a claim to search.")
        else:
            with st.spinner(f"Executing {method}..."):
                try:
                    results = execute_search(
                        query, method, top_k, corpus_df, bm25, dense_model, faiss_index, azure_client, cross_encoder
                    )
                    
                    if not results:
                        st.info("No documents found for this query.")
                    else:
                        st.success(f"Successfully retrieved Top {len(results)} matches.")
                        for idx, res in enumerate(results, 1):
                            with st.expander(f"#{idx} | Score: {res['score']:.4f} | {res['title']}"):
                                st.caption(f"**Doc ID:** `{res['doc_id']}`")
                                st.markdown(f"**Abstract:**\n\n{res['abstract']}")
                except Exception as e:
                    st.error(str(e))

# ==========================================
# TAB 2: COMPARE ALL METHODS
# ==========================================
with tab_compare:
    st.subheader("Algorithm Showdown")
    st.markdown("Enter a claim below. We will push it through **all 7 algorithms** simultaneously and display their absolute **#1 top prediction** side-by-side.")
    
    comp_query = st.text_input("SciFact Claim to evaluate:", "COVID-19 alters the expression of ACE2 in lung tissue.")
    compare_button = st.button("⚖️ Compare All Methods")
    
    if compare_button:
        if not comp_query.strip():
            st.error("Please enter a claim.")
        else:
            with st.spinner("Processing through 7 different retrieval pathways..."):
                cols = st.columns(3)
                col_idx = 0
                
                for m in ALL_METHODS:
                    try:
                        res = execute_search(
                            comp_query, m, 1, corpus_df, bm25, dense_model, faiss_index, azure_client, cross_encoder
                        )
                        target_col = cols[col_idx % 3]
                        with target_col:
                            st.info(f"**{m}**")
                            if res:
                                top_doc = res[0]
                                st.markdown(f"**Doc ID:** `{top_doc['doc_id']}`")
                                st.markdown(f"**Title:** {top_doc['title']}")
                                # Trim abstract for UI neatness
                                abstract_snippet = top_doc['abstract'][:250] + ("..." if len(top_doc['abstract']) > 250 else "")
                                st.caption(f"*\"{abstract_snippet}\"*")
                            else:
                                st.warning("No Result.")
                        col_idx += 1
                        
                    except Exception as e:
                        target_col = cols[col_idx % 3]
                        with target_col:
                            st.error(f"**{m}** failed.\n\n`{e}`")
                        col_idx += 1

# ==========================================
# TAB 3: METRICS DASHBOARD
# ==========================================
with tab_metrics:
    st.subheader("Performance Analytics")
    
    col_csv, col_img = st.columns([1, 1.5])
    
    with col_csv:
        csv_path = os.path.join("results", "final_results.csv")
        if os.path.exists(csv_path):
            st.markdown("#### Evaluation Table")
            df = pd.read_csv(csv_path)
            # Highlight max values in dataframe strictly for numeric columns
            st.dataframe(df.style.highlight_max(axis=0, color='lightgreen', subset=["Recall@1", "Recall@5", "Recall@10", "MRR@10"]))
        else:
            st.warning("`results/final_results.csv` not found. Please run the evaluation scripts first.")

    with col_img:
        img_path = os.path.join("results", "final_comparison.png")
        if os.path.exists(img_path):
            st.markdown("#### Recall & MRR Trends")
            img = Image.open(img_path)
            st.image(img, use_column_width=True)
        else:
            st.warning("`results/final_comparison.png` not found. Please run `plot_final_results.py`.")
