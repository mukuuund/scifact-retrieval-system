# SciFact Retrieval System: Analysis & Insights

## 1. Problem Statement
Scientific claim verification requires identifying exact sentences within thousands of scientific papers that support or refute a given claim. Standard search engines struggle with this because scientific claims involve dense domain-specific terminology, complex semantics, and subtle negations. The core challenge is building a pipeline that accurately retrieves a handful of heavily relevant documents out of a massive corpus without sacrificing speed.

## 2. Methods Compared
We progressively evaluated five distinct retrieval architectures:
- **BM25 (Local)**: A standard lexical search mechanism operating on exact keyword overlap.
- **Dense Retrieval (Local)**: Bi-encoder (`all-MiniLM-L6-v2`) embeddings mapping documents and queries into a shared vector space, indexed by FAISS.
- **Hybrid Retrieval (Local)**: A Reciprocal Rank Fusion (RRF) approach combining the exact-match precision of BM25 with the semantic understanding of the Dense model.
- **Azure AI Search (Cloud)**: A cloud-native implementation utilizing Azure's lexical matching, native HNSW vector integration, and cloud-scale hybrid querying.
- **Hybrid + Reranker (Advanced)**: Azure Hybrid search acting as a first-stage retriever, followed by a second-stage `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker evaluating pairwise text semantics.

## 3. Key Results Summary
- **Baseline Lexical models** fell behind instantly, struggling against vocabulary mismatches.
- **Dense Vector Search** provided a massive jump in Recall, capturing documents conceptually related to the claims.
- **Hybrid Models** safely captured the best of both worlds, pushing Recall@5 aggressively higher.
- **Azure's Cloud Hybrid Integration** outperformed custom local hybrid implementations, due to Microsoft's highly optimized internal normalization and lexical tooling.
- **Second-Stage Reranking** cemented the pipeline, taking the already high-quality Azure Hybrid output and reorganizing it semantically. This massively boosted **MRR@10**, pushing the absolute best documents flawlessly to the rank #1 position.

## 4. Key Insights
- **Why hybrid outperforms BM25 and dense alone:** BM25 fails when synonyms are heavily used, while Dense retrieval occasionally hallucinates relevance on vaguely related topics. Hybrid fusion ensures that a document is penalized unless it matches BOTH conceptually (Vector) and factually (Lexical).
- **Why Azure hybrid is better than local hybrid:** Local RRF (Reciprocal Rank Fusion) treats both signals uniformly. Azure AI Search uses proprietary internal scoring standardizations that balance BM25 weights against HNSW distance scores more smoothly across edge cases.
- **Why reranking improves MRR significantly:** Bi-encoders (used in Dense/Hybrid) compress entire paragraphs into single vectors independent of the query, losing crucial cross-attention interactions. Cross-Encoders pass the query *and* the document through the Transformer layers together, allowing deep semantic mapping at the cost of speed. Applying the Cross-Encoder only to the top 10 Hybrid results gives us this accuracy without the latency penalty.
- **Tradeoff between recall and precision:** First-stage retrievers prioritize Recall (finding *any* relevant document in the top 100). Second-stage rerankers prioritize Precision (pushing the *most* relevant documents to the top 1). Reranking doesn't always improve Recall@10 significantly, but it intensely improves MRR (Precision).

## 5. Failure Analysis Insights
- **When BM25 fails:** When a claim uses completely different phrasing than the document (e.g., "Cardiac arrest" vs "Heart failure").
- **When dense retrieval fails:** When exact numeric constraints or rare acronyms determine the truth value. Dense vectors effectively blur "15mg" and "50mg" into similar locations in hyperspace.
- **When hybrid still fails:** When the supporting evidence is buried deeply inside a very long document. Since we encode the "abstract", deeply buried findings within the full paper body can be missed.

## 6. Final Takeaway
Modern retrieval systems **must** be multi-stage. Relying solely on Vector Databases is a common anti-pattern that leads to confident failures on localized entity-heavy queries. A production-grade pipeline requires **Lexical Search for factual grounding**, **Vector Search for semantic intent**, and **Cross-Encoder Reranking for final cognitive alignment**.
