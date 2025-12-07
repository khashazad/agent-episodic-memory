#!/usr/bin/env python3

import os
from typing import Dict, List, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)


class ChromaClient:
    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "episodic_memory"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._connected = False

    def connect(self) -> bool:
        try:
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                settings=Settings(allow_reset=False)
            )

            # Get existing collection
            self.collection = self.client.get_collection(name=self.collection_name)
            self._connected = True
            logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            self._connected = False
            return False

    def is_connected(self) -> bool:
        return self._connected and self.collection is not None

    def find_similar_actions(self, query_embedding: np.ndarray,
                           n_results: int = 5) -> List[Dict]:

        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            # Prepare query
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": n_results,
                "include": ["documents", "distances"]
            }

            results = self.collection.query(**query_params)

            similar_actions = []

            for i in range(len(results["documents"][0])):
                similar_actions.append({
                    "document": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1.0 - results["distances"][0][i]  # Convert distance to similarity
                })

            return similar_actions

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

def create_client_from_env() -> ChromaClient:
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    collection = os.getenv("CHROMA_COLLECTION", "episodic_memory")

    return ChromaClient(host=host, port=port, collection_name=collection)
