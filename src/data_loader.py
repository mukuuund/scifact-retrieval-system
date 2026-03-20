"""
Module for loading the SciFact corpus and claims datasets.
"""

import json
from typing import List, Dict, Any
import pandas as pd


def load_corpus(file_path: str) -> pd.DataFrame:
    """
    Load the corpus from a JSONL file and create a combined text field.
    
    Args:
        file_path (str): Path to the corpus.jsonl file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the corpus with a 'combined_text' column.
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
            
    df = pd.DataFrame(records)
    
    # Ensure doc_id is an integer for consistent matching
    if 'doc_id' in df.columns:
        df['doc_id'] = df['doc_id'].astype(int)
    
    # Create combined text from title and abstract
    if 'title' in df.columns and 'abstract' in df.columns:
        # In SciFact, 'abstract' is typically a list of sentences
        df['abstract_text'] = df['abstract'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else str(x)
        )
        df['abs_text'] = df['abstract_text']  # Alias for dense retrieval
        
        # Combine title and abstract
        df['combined_text'] = df['title'] + " " + df['abstract_text']
        df['text'] = df['combined_text']  # Alias for dense retrieval
        
    return df


def load_claims(file_path: str) -> pd.DataFrame:
    """
    Load claims from a JSONL file and extract relevant documents for evaluation.
    
    Args:
        file_path (str): Path to the claims JSONL file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the claims and their relevant doc_ids.
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Extract relevant doc_ids if evidence is present
            relevant_docs = []
            if 'evidence' in data:
                # evidence is a dict mapping doc_id string -> list of evidence lists
                relevant_docs = list(data['evidence'].keys())
                
            # Store as integers for consistent matching with corpus
            doc_ids = [int(doc_id) for doc_id in relevant_docs]
            data['relevant_doc_ids'] = doc_ids
            data['cited_doc_ids'] = doc_ids  # Alias for dense retrieval
            records.append(data)
            
    return pd.DataFrame(records)
