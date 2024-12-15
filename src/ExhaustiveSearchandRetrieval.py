import os
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Document and Query Processing
def get_documents_and_document_name(doc_folder):
    documents = []
    doc_names = []
    for file in os.listdir(doc_folder):
        if file.endswith(".txt"):
            with open(os.path.join(doc_folder, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
            doc_names.append(file.replace("output_", "").replace(".txt", ""))
    return documents, doc_names


def read_query_and_query_numbers(query_file):
    queries_df = pd.read_excel(query_file)
    queries = queries_df["Query"].tolist()
    query_numbers = queries_df["Query number"].tolist()
    return queries, query_numbers


# Embedding Preparation
def document_and_query_embedding_preparation(embedding_model_name, documents, queries):
    model = SentenceTransformer(embedding_model_name)
    doc_embeddings = model.encode(documents)
    query_embeddings = model.encode(queries)
    return doc_embeddings, query_embeddings


# Semantic Search
def semantic_search(query_embeddings, doc_embeddings, doc_names, query_numbers, top_k=10):
    results = []
    for query_idx, query_embedding in enumerate(query_embeddings):
        similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
        ranked_indices = np.argsort(similarities)[::-1]
        top_k_docs = [doc_names[idx] for idx in ranked_indices[:top_k]]
        for doc in top_k_docs:
            results.append({
                "Query number": query_numbers[query_idx],
                "Similar Doc": doc,
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_excel("semantic_search_results.xlsx", index=False)
    return results_df


def map_query_to_documents(dev_query_results_file, results_df):
    dev_query_results_df = pd.read_csv(dev_query_results_file)
    dev_query_results = (
        dev_query_results_df.groupby("Query_number")["doc_number"]
        .apply(list)
        .to_dict()
    )

    results_grouped = (
        results_df.groupby("Query number")["Similar Doc"]
        .apply(list)
        .to_dict()
    )

    return {
        "dev_query_results": dev_query_results,
        "results_grouped": results_grouped,
    }


# Precision and Recall Calculation
def precision_at_k(retrieved, relevant, k):
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_at_k & relevant_set) / k


def recall_at_k(retrieved, relevant, k):
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_at_k & relevant_set) / len(relevant_set)


# Evaluation
def evaluate_retrieval(results_grouped, dev_query_results, k_values=[1, 3, 5, 10]):
    mean_precision_at_k = {k: [] for k in k_values}
    mean_recall_at_k = {k: [] for k in k_values}

    # Evaluate each query
    for query_number, retrieved_docs in results_grouped.items():
        relevant_docs = [str(doc_number) for doc_number in dev_query_results.get(query_number, [])]  # Ground truth docs
        for k in k_values:
            if k <= len(retrieved_docs):
                mean_precision_at_k[k].append(precision_at_k(retrieved_docs, relevant_docs, k))
                mean_recall_at_k[k].append(recall_at_k(retrieved_docs, relevant_docs, k))

    # Calculate mean precision and recall for each k
    for k in k_values:
        mean_precision_at_k[k] = np.mean(mean_precision_at_k[k]) if mean_precision_at_k[k] else 0
        mean_recall_at_k[k] = np.mean(mean_recall_at_k[k]) if mean_recall_at_k[k] else 0

    # Display evaluation metrics
    print("Mean Precision@k:", mean_precision_at_k)
    print("Mean Recall@k:", mean_recall_at_k)

    return {
        "mean_precision_at_k": mean_precision_at_k,
        "mean_recall_at_k": mean_recall_at_k,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_folder", type=str, default="./SmallDataset/full_docs_small",
                        help="Documents Folder")
    parser.add_argument("--query_file", type=str, default="./SmallDataSet/dev_small_queries.xlsx",
                        help="Queries Path")
    parser.add_argument("--dev_query_results_file", type=str, default="./SmallDataSet/dev_query_results_small.csv",
                        help="Queries Document Matching csv")
    parser.add_argument("--embedding_model_name", type=str, default="all-MiniLM-L6-v2",
                        help="Embedding Model Name")

    args = parser.parse_args()

    start = time.time()

    # Read Documents and Queries
    documents, doc_names = get_documents_and_document_name(args.doc_folder)
    queries, query_numbers = read_query_and_query_numbers(args.query_file)
    print("Files Loaded...")
    print("Document are Loaded in ", round(time.time() - start, 2), "seconds\n")

    embedding_start_time = time.time()
    # Create Embeddings
    doc_embeddings, query_embeddings = document_and_query_embedding_preparation(
        args.embedding_model_name, documents, queries
    )
    print("Embeddings Generated...\n")

    # Perform Semantic Search
    results_df = semantic_search(query_embeddings, doc_embeddings, doc_names, query_numbers)
    print("Semantic Search Run...\n")

    # Map and Get Results
    mappings = map_query_to_documents(args.dev_query_results_file, results_df)
    dev_query_results = mappings["dev_query_results"]
    results_grouped = mappings["results_grouped"]
    print("Query Mapped to Documents...")
    print("Embedding and Semantic Search Run Time: ", round(time.time() - embedding_start_time, 2), "seconds\n")

    # Evaluate Retrieval
    print("Retrieval Evaluating...\n")
    k_values = [1, 3, 5, 10]
    evaluation_metrics = evaluate_retrieval(results_grouped, dev_query_results, k_values)
    print("Total Run Time: ", round(time.time() - start, 2), "seconds")
