# Pylon Issue Clustering and Analysis

This project analyzes customer issues from Pylon, clusters them, and generates titles for each cluster using AI. It utilizes ChromaDB for vector storage and querying, and Ollama OSS models for text embedding and cluster title generation.

## Features

1. Fetches recent issues from Pylon API
2. Creates embeddings for issues using Nomic Embed Text model (Ollama)
3. Stores issue embeddings in a persistent ChromaDB collection
4. Clusters issues using K-means algorithm
5. Generates titles for each cluster using Llama 3.2
6. Option to semantically search for issues using Chroma's built in similarity search

## Requirements

- ChromaDB
- Ollama

## Setup

1. Install the required packages:
   ```
   pip3 install -r requirements.txt
   ```

2. Ensure Ollama is running locally with the `nomic-embed-text` and `llama3.2:latest` models downloaded.

3. Set up your Pylon API key in `.env` file

## Usage

Run the script with:

```
python3 run.py
```