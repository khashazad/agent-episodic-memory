#!/usr/bin/env python3

import os
from typing import Dict, List, Optional
import numpy as np
import logging

from pymilvus import connections, Collection, utility

logger = logging.getLogger(__name__)

class MilvusClient:
    """
    Minimal Milvus analog of your ChromaClient.

    Assumptions:
      - Collection schema includes:
          id: VARCHAR (primary key)
          embedding: FLOAT_VECTOR
          document: VARCHAR
      - You created an index already (HNSW / IVF_PQ / DISKANN) and loaded the collection.
    """

    def __init__(self, host: str = "localhost", port: int = 19530, collection_name: str = "episodic_memory_hnsw",
                 alias: str = "default"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.alias = alias

        self.collection: Optional[Collection] = None
        self._connected = False

    def connect(self) -> bool:
        try:
            connections.connect(alias=self.alias, host=self.host, port=str(self.port))

            if not utility.has_collection(self.collection_name, using=self.alias):
                raise RuntimeError(f"Milvus collection '{self.collection_name}' does not exist")

            self.collection = Collection(self.collection_name, using=self.alias)

            # Load into memory for search (important)
            self.collection.load()

            self._connected = True
            logger.info(f"Connected to Milvus collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self._connected = False
            self.collection = None
            return False

    def is_connected(self) -> bool:
        return self._connected and self.collection is not None

    @staticmethod
    def _milvus_distance_to_similarity(distance: float, metric_type: str) -> float:
        """
        Chroma returns a distance; Milvus returns a 'score' that depends on metric.
        This helper tries to give you a consistent "similarity-ish" number.
        """
        m = metric_type.upper()
        if m == "L2":
            # Not a true cosine similarity; this is a monotonic transform in [0,1] for non-negative distances.
            return 1.0 / (1.0 + float(distance))
        if m in ("IP", "COSINE"):
            # For IP/COSINE, higher is already "more similar".
            return float(distance)
        return float(distance)

    def find_similar_actions(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        metric_type: str = "L2",
        search_params: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        search_params examples:
          - HNSW:    {"ef": 64}
          - IVF_*:   {"nprobe": 16}
          - DISKANN: {"search_list": 64}   (name may vary by version/config)

        If you pass None, we'll pick reasonable defaults based on collection index type isn't available here,
        so we default to IVF-ish safe params.
        """
        if not self.is_connected():
            logger.error("Not connected to Milvus")
            return []

        try:
            emb = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
            vectors = [emb.tolist()]

            if search_params is None:
                # Reasonable "works most places" default. For HNSW you probably want ef.
                search_params = {"nprobe": 16}

            results = self.collection.search(
                data=vectors,
                anns_field="embedding",
                param=search_params,
                limit=n_results,
                output_fields=["document"],
                consistency_level="Strong",
            )

            similar_actions: List[Dict] = []
            hits = results[0]

            for hit in hits:
                # hit.distance is "distance" for L2, "score" for IP/COSINE depending on metric.
                dist = float(hit.distance)
                doc = hit.entity.get("document") if hit.entity is not None else None

                similar_actions.append({
                    "id": hit.id,
                    "document": doc,
                    "distance": dist,
                    "similarity": self._milvus_distance_to_similarity(dist, metric_type),
                })

            return similar_actions

        except Exception as e:
            logger.error(f"Milvus query failed: {e}")
            return []


def create_client_from_env() -> MilvusClient:
    host = os.getenv("MILVUS_HOST", "localhost")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    collection = os.getenv("MILVUS_COLLECTION", "episodic_memory")
    alias = os.getenv("MILVUS_ALIAS", "default")

    return MilvusClient(host=host, port=port, collection_name=collection, alias=alias)