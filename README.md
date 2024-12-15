# Semantic Search and Retrieval System
## Project Overview

This project consists of two scripts to implement a semantic search and retrieval system using Sentence-BERT embeddings. Each script is tailored for a specific use case:

**ExhaustiveSearchandRetrieval.py:** Implements an exhaustive semantic search system for smaller datasets.
**EfficientSearchandRetrievalUsingInvertedIndexing.py:** Scales the search system for large datasets using clustering and inverted indexing for efficiency.

## Implementation Details
**1. ExhaustiveSearchandRetrieval.py**
- Designed for small datasets.
- Converts all documents into vector embeddings using Sentence-BERT.
For each query:
- Encodes the query into a vector.
- Computes cosine similarity between the query vector and all document embeddings.
- Ranks documents by similarity scores and returns the top-10 most relevant results.
- Results are evaluated using Precision@k and Recall@k for k = 1, 3, 5, 10.

**2. EfficientSearchandRetrievalUsingInvertedIndexing.py**
- Optimized for large datasets with over 100,000 documents.
- Implements clustering and inverted indexing:
- Applies KMeans clustering to group document embeddings into k clusters.
- Constructs an inverted index mapping each cluster to its document IDs.
For each query:
- Computes cosine similarity with cluster centroids to select the top-k clusters.
- Searches within the selected clusters and ranks documents by similarity.
- Returns the top-10 most relevant documents.
- Precision and recall metrics are calculated to evaluate retrieval effectiveness.
- Supports configurable cluster counts to optimize runtime and performance.

## Features
**ExhaustiveSearchandRetrieval.py:**
Suitable for smaller datasets.
Straightforward implementation for semantic similarity search.

**EfficientSearchandRetrievalUsingInvertedIndexing.py:**
Scalable for large datasets.
Incorporates clustering and inverted indexing for reduced runtime and efficient retrieval.

## Usage
**1. ExhaustiveSearchandRetrieval.py**

Run this script for exhaustive search on small datasets:
python script_name.py --doc_folder "docs" \
                      --query_file "queries.xlsx" \
                      --dev_query_results_file "results.csv" \
                      --embedding_model_name "all-MiniLM-L12-v2"

**2. EfficientSearchandRetrievalUsingInvertedIndexing.py**

Run this script for efficient search on large datasets:
python EfficientSearchandRetrievalUsingInvertedIndexing.py \
    --doc_folder "largedocs" \
    --query_file dev_queries.tsv \
    --dev_query_results_file dev_query_results.csv \
    --embeddings_file large_doc_embeddings.npy \
    --num_clusters 20 \
    --model_name all-MiniLM-L6-v2

## Testing Queries
After completing the embedding generation, clustering, and inverted indexing process, the system allows for seamless testing of queries. Users can provide their own queries in the specified format and evaluate the system's performance using precision and recall metrics. The **Test.py** script supports query testing by performing semantic search against the indexed documents and generating detailed evaluation metrics for specified k values. This feature ensures the system's flexibility and usability for real-world scenarios, enabling quick validation of the retrieval performance.

## Performance Evaluation
**Precision and Recall:**
- Both scripts calculate Precision@k and Recall@k to measure the accuracy and completeness of retrieval results.

**Runtime Efficiency:**
- ExhaustiveSearchandRetrieval.py has higher computational overhead for large datasets due to pairwise comparisons.
- EfficientSearchandRetrievalUsingInvertedIndexing.py reduces runtime linearly by increasing cluster counts, optimizing search efficiency for large datasets.

## Conclusion
This system offers two tailored solutions for semantic search:

**ExhaustiveSearchandRetrieval.py** is ideal for smaller datasets with straightforward exhaustive comparisons.
**EfficientSearchandRetrievalUsingInvertedIndexing.py** scales seamlessly to large datasets by leveraging clustering and inverted indexing. Both scripts provide flexibility, efficiency, and meaningful results using Sentence-BERT embeddings.
