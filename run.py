from datetime import datetime, timedelta
import requests
import chromadb
from chromadb.utils import embedding_functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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

def semantic_search(query, collection, n_results=2):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    except Exception as e:
        logging.error(f"Error performing semantic search: {e}")
        return None

def generate_cluster_title(cluster_issues):
    try:
        r = requests.post('http://localhost:11434/api/generate', json={
            'model': 'llama3.2:latest',
            'prompt': f"Generate one title for the following cluster of issues. Return the title and nothing else. Issues: {cluster_issues}",
            'stream': False
        })
        r.raise_for_status()
        return r.json()['response']
    except requests.RequestException as e:
        logging.error(f"Error generating cluster title: {e}")
        return "Untitled Cluster"

def cluster_issues(collection, no_of_clusters=10):
    try:
        # Get all documents from the collection
        documents = collection.get()['documents']

        # Vectorize the documents
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(documents)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=no_of_clusters, random_state=42)
        kmeans.fit(X)

        # Get the cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Get the cluster labels
        cluster_labels = kmeans.labels_

        # Create a dictionary to store issues by cluster
        clustered_issues = {i: [] for i in range(no_of_clusters)}

        # Assign issues to their respective clusters
        for i, label in enumerate(cluster_labels):
            clustered_issues[label].append(documents[i])

        # Create titles for each cluster using Llama 3.2 8b
        cluster_titles = []
        for cluster_id in clustered_issues:
            cluster_issues = clustered_issues[cluster_id]
            cluster_titles.append(generate_cluster_title(cluster_issues))

        return clustered_issues, cluster_titles
    except Exception as e:
        logging.error(f"Error clustering issues: {e}")
        return {}, []

if __name__ == "__main__":
    try:
        collection_name = "pylon_issues"
        collection = client.get_or_create_collection(collection_name, embedding_function=embedding_function)

        # issues = get_pylon_issues(days_ago=29)
        # issues = filter_issues(issues)
        # create_issue_embeddings(issues, collection)

        # query = "api pricing"
        # logging.info(f'QUERY: {query}')
        # results = semantic_search(query, collection)
        # if results:
        #     for r in results['documents'][0]:
        #         print(r)
        #         print()

        clustered_issues, cluster_titles = cluster_issues(collection, no_of_clusters=10)
        for cluster_title, issues in clustered_issues.items():
            print(cluster_titles[cluster_title])
            for issue in issues:
                print(issue + '\n')
            print('\n\n')
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")