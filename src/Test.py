import argparse
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm

def load_doc_names(doc_folder):
    doc_names = []
    for file in tqdm(os.listdir(doc_folder), desc="Loading Document Names..."):
        if file.endswith(".txt"):
            doc_names.append(file.replace("output_", "").replace(".txt", ""))
    return doc_names

def search_documents(query, model, cluster_centroids, inverted_index, embeddings, doc_names, top_k_clusters=3):
    query_vector = model.encode([query])[0]
    similarities = cosine_similarity([query_vector], cluster_centroids)[0]
    top_clusters = np.argsort(similarities)[::-1][:top_k_clusters]
    
    candidate_docs = []
    for cluster in top_clusters:
        candidate_docs.extend(inverted_index[cluster])
    candidate_docs = list(set(candidate_docs))
    candidate_embeddings = [embeddings[doc_id] for doc_id in candidate_docs]
    
    doc_similarities = cosine_similarity([query_vector], candidate_embeddings)[0]
    ranked_docs = sorted(zip(candidate_docs, doc_similarities), key=lambda x: x[1], reverse=True)
    return ranked_docs

def main(args):
    # Loading or Creating Document Name 
    if os.path.exists(args.doc_names_path):
        print(f"{args.doc_names_path} Exists...")
        with open(args.doc_names_path, "rb") as f:
            doc_names = pickle.load(f)
    else:
        print(f"{args.doc_names_path} Does Not Exist...")
        doc_names = load_doc_names(args.doc_folder)
        with open(args.doc_names_path, "wb") as f:
            pickle.dump(doc_names, f)
        print(f"{args.doc_names_path} Created")

    # Load Embeddings, Cluster Centroids, Inverted Index (For Testing)
    embeddings = np.load(args.embeddings_path)
    cluster_centroids = np.load(args.cluster_centroids_path)
    with open(args.inverted_index_path, "rb") as f:
        inverted_index = pickle.load(f)

    # Loading Test Queries
    queries_df = pd.read_csv(args.query_file, sep='\t')
    queries = queries_df["Query"].tolist()
    query_numbers = queries_df["Query number"].tolist()

    # Loading Embedding Model
    embedding_model = SentenceTransformer(args.model_name)

    results = []
    for query, query_number in tqdm(zip(queries, query_numbers), desc="Processing Queries..."):
        ranked_docs = search_documents(query, embedding_model, cluster_centroids, inverted_index, embeddings, doc_names, top_k_clusters=3)
        top_10_docs = [doc_names[doc_id] for doc_id, _ in ranked_docs[:10]]
        for doc_number in top_10_docs:
            results.append({"Query_number": query_number, "doc_number": doc_number})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
    print("Results saved to results.csv.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", type=str, default="../LargeDataset/queries.csv", help="Test Query File")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer Embedding model")
    parser.add_argument("--doc_folder", type=str, default="../LargeDataset/full_docs/full_docs", help="Documents Folder to Get Document Name")
    parser.add_argument("--embeddings_path", type=str, default="large_doc_embeddings.npy", help="Embeddings .npy File")
    parser.add_argument("--cluster_centroids_path", type=str, default="cluster_centroids_100.npy", help="Cluster Centroids .npy File")
    parser.add_argument("--inverted_index_path", type=str, default="inverted_index_100.pkl", help="Inverted Index .pkl File")
    parser.add_argument("--doc_names_path", type=str, default="doc_names.pkl", help="Doc_names .pkl File")
    args = parser.parse_args()

    main(args)
