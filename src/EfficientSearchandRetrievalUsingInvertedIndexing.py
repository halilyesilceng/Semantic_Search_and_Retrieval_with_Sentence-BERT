import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

def load_documents(doc_folder):
    documents = []
    doc_names = []
    for file in tqdm(os.listdir(doc_folder), desc="Loading Documents..."):
        if file.endswith(".txt"):
            with open(os.path.join(doc_folder, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
            doc_names.append(file.replace("output_", "").replace(".txt", ""))
    return documents, doc_names

def embed_documents(documents, model_name='all-MiniLM-L6-v2', batch_size=64):
    model = SentenceTransformer(model_name)
    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding Documents..."):
        batch = documents[i:i + batch_size]
        embeddings.extend(model.encode(batch))
    return np.array(embeddings), model

def create_cluster_and_build_inverted_index(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    cluster_centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    inverted_index = {i: [] for i in range(num_clusters)}
    for doc_id, label in enumerate(cluster_labels):
        inverted_index[label].append(doc_id)
    
    return inverted_index, cluster_centroids, kmeans

def search_documents(query, model, cluster_centroids, inverted_index, embeddings, top_k_clusters=3):
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

def evaluate_results(queries, query_numbers, dev_query_results_df, model, cluster_centroids, inverted_index, embeddings, doc_names, k_values=[1, 3, 5, 10]):
    precisions = {k: [] for k in k_values}
    recalls = {k: [] for k in k_values}

    for query, query_number in tqdm(zip(queries, query_numbers), desc="Evaluation of Queries..."):
        relevant_docs = dev_query_results_df.loc[dev_query_results_df["Query_number"] == query_number]["doc_number"].tolist()
        ranked_docs = search_documents(query, model, cluster_centroids, inverted_index, embeddings, top_k_clusters=3)
        retrieved_docs = [int(doc_names[doc_id]) for doc_id, _ in ranked_docs[:max(k_values)]]

        for k in k_values:
            top_k_retrieved = retrieved_docs[:k]
            relevant_retrieved = len(set(top_k_retrieved) & set(relevant_docs))
            precisions[k].append(relevant_retrieved / k)
            recalls[k].append(relevant_retrieved / len(relevant_docs) if relevant_docs else 0)

    mean_precisions = {k: np.mean(precisions[k]) for k in k_values}
    mean_recalls = {k: np.mean(recalls[k]) for k in k_values}
    return mean_precisions, mean_recalls

def main(args):
    # Loading documents
    documents, doc_names = load_documents(args.doc_folder)
    # Save doc_names for testing
    with open("doc_names.pkl", "wb") as f:
        pickle.dump(doc_names, f)
    # Loading or Creating Embeddings
    if args.embeddings_file and os.path.exists(args.embeddings_file):
        embeddings = np.load(args.embeddings_file)
        print("Loaded embeddings from file.")
    else:
        embeddings, embedding_model = embed_documents(documents, args.model_name)
        if args.embeddings_file:
            np.save(args.embeddings_file, embeddings)
            print("Saved embeddings to file.")

    # Creating Clustering and inverted index
    inverted_index, cluster_centroids, _ = create_cluster_and_build_inverted_index(embeddings, args.num_clusters)

    # Saving Inverted Index and Centroids for Later Use
    with open("inverted_index.pkl", "wb") as f:
        pickle.dump(inverted_index, f)
    np.save("cluster_centroids.npy", cluster_centroids)

    # Loading First 1000 Queries and dev_query_results
    queries_df = pd.read_csv(args.query_file, sep="\t")
    queries = queries_df["Query"].tolist()[:1000]
    query_numbers = queries_df["Query number"].tolist()[:1000]
    dev_query_results_df = pd.read_csv(args.dev_query_results_file)

    # Evaluation of Results
    embedding_model = SentenceTransformer(args.model_name)
    k_values = [1, 3, 5, 10]
    mean_precisions, mean_recalls = evaluate_results(
        queries, 
        query_numbers,
        dev_query_results_df, 
        embedding_model, 
        cluster_centroids, 
        inverted_index, 
        embeddings, 
        doc_names, 
        k_values
    )

    print("Mean Precision at k:", mean_precisions)
    print("Mean Recall at k:", mean_recalls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_folder", type=str, default="../LargeDataset/full_docs/full_docs", help="Documents Path")
    parser.add_argument("--query_file", type=str, default="../LargeDataset/dev_queries.tsv", help="Query Path")
    parser.add_argument("--dev_query_results_file", type=str, default="../LargeDataset/dev_query_results.csv", help="Document which Keep Query and Document Match")
    parser.add_argument("--embeddings_file", type=str, default="large_doc_embeddings.npy", help="Embedding Path")
    parser.add_argument("--num_clusters", type=int, default=10, help="KMeans Cluster Counts")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="Embedding Models")
    args = parser.parse_args()

    main(args)