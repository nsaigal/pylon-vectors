from datetime import datetime, timedelta
import requests
import chromadb
from chromadb.utils import embedding_functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Init Chroma client
client = chromadb.Client()

# Create a persistent collection for the issues in the local folder
client = chromadb.PersistentClient(path="./pylon_issues.db")

# Option 1: use SentenceTransformer for embedding
# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device='cpu')

# Option 2: use Ollama for embedding
embedding_function = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)

def get_date_range(days_ago):
    from_date = datetime.now() - timedelta(days=days_ago)
    to_date = datetime.now()

    # format RFC 3339
    rfc_3339_format_start = from_date.isoformat() + 'Z'
    rfc_3339_format_end = to_date.isoformat() + 'Z'

    return rfc_3339_format_start, rfc_3339_format_end

def get_pylon_issues(days_ago=1):
    try:
        start_date, end_date = get_date_range(days_ago)
        r = requests.get(
            f"https://api.usepylon.com/issues?start_time={start_date}&end_time={end_date}", 
            headers={"Authorization": f"Bearer {os.getenv('PYLON_API_KEY')}"}
        )
        r.raise_for_status()  # Raise an exception for bad status codes
        return r.json()['data']
    except requests.RequestException as e:
        logging.error(f"Error fetching Pylon issues: {e}")
        return []

def filter_issues(issues):
    issues = [i for i in issues if len(i['body_html']) > 50]
    return issues[0:10]

def create_issue_embeddings(issues, collection):
    logging.info('Creating embeddings...')
    try:
        for i, issue in enumerate(issues):
            # Here we adding our custom issue fields to the embedding metadata, feel free to remove these or your own
            product_areas = issue.get('custom_fields', {}).get('product_area', {}).get('values', [])
            product_areas_first = product_areas[0] if product_areas else ""
            product_lines = issue.get('custom_fields', {}).get('product_line', {}).get('values', [])
            product_lines_joined = ", ".join(product_lines)
            
            collection.upsert(
                documents=[issue['body_html']],
                metadatas=[{"issue_id": issue['id'], "title": issue['title'], "created_at": issue['created_at'], "product_areas": product_areas_first, "product_lines": product_lines_joined}],
                ids=[str(issue['id'])]
            )
        logging.info(f"Successfully created embeddings for {len(issues)} issues")
    except Exception as e:
        logging.error(f"Error creating issue embeddings: {e}")

def semantic_search(query, collection, n_results=10):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    except Exception as e:
        logging.error(f"Error performing semantic search: {e}")
        return None

def generate_cluster_title(cluster_metadata):
    try:
        # Prepare a summary of the cluster for better title generation
        issue_summaries = [f"Title: {meta['title']}, Product Area: {meta['product_areas']}" 
                           for meta in cluster_metadata[:5]]  # Limit to first 5 for brevity
        cluster_summary = "\n".join(issue_summaries)

        r = requests.post('http://localhost:11434/api/generate', json={
            'model': 'llama3.2:latest',
            'prompt': f"Generate a concise and descriptive title for the following cluster of issues. The title should capture the main theme or problem addressed in these issues. Cluster summary:\n{cluster_summary}\n\nTitle:",
            'stream': False
        })
        r.raise_for_status()
        return r.json()['response'].strip()
    except requests.RequestException as e:
        logging.error(f"Error generating cluster title: {e}")
        return "Untitled Cluster"

def cluster_issues(collection, min_clusters=2, max_clusters=10):
    try:
        # Get all documents and their metadata from the collection
        result = collection.get()
        documents = result['documents']
        metadatas = result['metadatas']

        # Ensure we have enough documents for clustering
        n_samples = len(documents)
        if n_samples < 2:
            logging.warning("Not enough samples for clustering. Returning all documents in a single cluster.")
            return {0: documents}, {0: metadatas}, ["All Issues"]

        # Adjust max_clusters if necessary
        max_clusters = min(max_clusters, n_samples - 1)
        min_clusters = min(min_clusters, max_clusters)

        # Vectorize the documents
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(documents)

        # Find the optimal number of clusters using silhouette score
        silhouette_scores = []
        K = range(min_clusters, max_clusters + 1)
        for k in K:
            if k >= n_samples:
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            if len(set(cluster_labels)) < 2:
                continue  # Skip if we ended up with only one cluster
            score = silhouette_score(X, cluster_labels)
            silhouette_scores.append(score)

        if not silhouette_scores:
            logging.warning("Could not compute valid clusters. Returning all documents in a single cluster.")
            return {0: documents}, {0: metadatas}, ["All Issues"]

        optimal_k = K[np.argmax(silhouette_scores)]

        # Perform K-means clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Create dictionaries to store issues and metadata by cluster
        clustered_issues = {i: [] for i in range(optimal_k)}
        clustered_metadata = {i: [] for i in range(optimal_k)}

        # Assign issues and metadata to their respective clusters
        for i, label in enumerate(cluster_labels):
            clustered_issues[label].append(documents[i])
            clustered_metadata[label].append(metadatas[i])

        # Create titles for each cluster
        cluster_titles = []
        for cluster_id in clustered_issues:
            cluster_issues = clustered_issues[cluster_id]
            cluster_meta = clustered_metadata[cluster_id]
            cluster_titles.append(generate_cluster_title(cluster_meta))

        return clustered_issues, clustered_metadata, cluster_titles
    except Exception as e:
        logging.error(f"Error clustering issues: {e}")
        return {0: documents}, {0: metadatas}, ["All Issues (Error in clustering)"]

if __name__ == "__main__":
    try:
        collection_name = "pylon_issues"
        collection = client.get_or_create_collection(collection_name, embedding_function=embedding_function)

        # issues = get_pylon_issues(days_ago=29)
        # issues = filter_issues(issues)
        # create_issue_embeddings(issues, collection)

        # query = "integration"
        # logging.info(f'QUERY: {query}')
        # results = semantic_search(query, collection)
        # if results:
        #     for r in results['documents'][0]:
        #         print(r)
        #         print()

        clustered_issues, clustered_metadata, cluster_titles = cluster_issues(collection, min_clusters=2, max_clusters=10)
        for i, (issues, metadata, title) in enumerate(zip(clustered_issues.values(), clustered_metadata.values(), cluster_titles)):
            print(f"Cluster {i+1}: {title}")
            print('Number of issues:', len(issues))
            for issue, meta in zip(issues[:3], metadata[:3]):  # Print first 3 issues of each cluster
                print(f"Title: {meta['title']}")
                print(f"Product Area: {meta['product_areas']}")
                print(f"Issue snippet: {issue[:100]}...")  # Print first 100 characters of each issue
                print()
            print('\n')
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")