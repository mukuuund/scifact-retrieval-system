"""
Module for BM25 retrieval on the SciFact corpus.
"""

from typing import List
import pandas as pd
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    A class to perform BM25 retrieval on a corpus.
    """
    
    def __init__(self, corpus_df: pd.DataFrame):
        """
        Initialize the retriever with the given corpus.
        
        Args:
            corpus_df (pd.DataFrame): DataFrame containing the corpus with 
                                      'doc_id' and 'combined_text' columns.
        """
        # Store document IDs mappings
        self.doc_ids = corpus_df['doc_id'].tolist()
        
        # Tokenize the documents (simple whitespace tokenization for baseline)
        # For a more advanced setup, you could use NLTK or spaCy here
        corpus_texts = corpus_df['combined_text'].fillna("").tolist()
        self.tokenized_corpus = [text.lower().split() for text in corpus_texts]
        
        # Initialize the BM25 model
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[int]:
        """
        Retrieve the top-k document IDs for a given query.
        
        Args:
            query (str): The search query (claim).
            top_k (int): The number of top documents to retrieve.
            
        Returns:
            List[int]: A list of the top-k retrieved doc_ids.
        """
        # Tokenize the query
        tokenized_query = query.lower().split()
        
        # Get scores for all documents
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get the indices of the top-k scores
        top_k_indices = sorted(
            range(len(doc_scores)), 
            key=lambda i: doc_scores[i], 
            reverse=True
        )[:top_k]
        
        # Map indices back to doc_ids
        top_k_doc_ids = [self.doc_ids[i] for i in top_k_indices]
        return top_k_doc_ids
