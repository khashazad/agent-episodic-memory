#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaDBSeeder:
    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8000):
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.client = None
        self.collection = None

    def connect(self) -> bool:
        try:
            self.client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=Settings(allow_reset=True)
            )

            # Test connection
            heartbeat = self.client.heartbeat()
            logger.info(f"Connected to ChromaDB: {heartbeat}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            return False

    # Create or get the collection for storing action sequences.
    def create_collection(self, collection_name: str = "episodic_memory", reset: bool = False):
        try:
            if reset and collection_name in [c.name for c in self.client.list_collections()]:
                self.client.delete_collection(collection_name)
                logger.info(f"Reset collection: {collection_name}")

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=None
            )

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise


    def load_chunk_data(self, chunk_dir: Path) -> Optional[Dict]:
        try:
            # Load embedding
            embedding_path = chunk_dir / "embedding.npy"
            if not embedding_path.exists():
                logger.warning(f"No embedding found in {chunk_dir}")
                return None

            embedding = np.load(embedding_path)

            # Load action descriptions
            actions_path = chunk_dir / "actions.txt"
            if not actions_path.exists():
                logger.warning(f"No actions.txt found in {chunk_dir}")
                return None

            with open(actions_path, 'r') as f:
                action_lines = f.read().strip().split('\n')

            # Clean up action descriptions
            action_descriptions = []
            for line in action_lines:
                if line.strip() and line.startswith('Step'):
                    # Extract description after "Step X: "
                    desc = line.split(': ', 1)[1] if ': ' in line else line
                    action_descriptions.append(desc)

            return {
                'embedding': embedding,
                'action_descriptions': action_descriptions,
            }

        except Exception as e:
            logger.error(f"Error loading chunk data from {chunk_dir}: {e}")
            return None

    """Get all chunk directories from the dataset."""
    def get_chunk_dirs(self, dataset_dir: Path) -> List[Path]:
        chunk_dirs = []

        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return chunk_dirs

        for episode_dir in dataset_dir.iterdir():
            if episode_dir.is_dir():
                for chunk_dir in episode_dir.iterdir():
                    if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk_"):
                        chunk_dirs.append(chunk_dir)

        return sorted(chunk_dirs)

    def seed_from_dataset(self, dataset_dir: Path, batch_size: int = 100):
        chunk_dirs = self.get_chunk_dirs(dataset_dir)

        if not chunk_dirs:
            logger.error(f"No chunks found in {dataset_dir}")
            return

        logger.info(f"Found {len(chunk_dirs)} chunks to process")

        # Process in batches
        processed = 0
        batch_embeddings = []
        batch_documents = []
        batch_ids = []

        for chunk_dir in tqdm(chunk_dirs, desc="Seeding ChromaDB"):
            chunk_data = self.load_chunk_data(chunk_dir)
            if chunk_data is None:
                continue

            # Prepare data for ChromaDB
            embedding = chunk_data['embedding'].tolist()  # Convert numpy to list
            document = "\n".join(chunk_data['action_descriptions'])

            # Create unique ID from directory path (e.g., "episode_001_chunk_003")
            episode_name = chunk_dir.parent.name
            chunk_name = chunk_dir.name
            chunk_id = f"{episode_name}_{chunk_name}"

            batch_embeddings.append(embedding)
            batch_documents.append(document)
            batch_ids.append(chunk_id)

            # Store batch when full
            if len(batch_embeddings) >= batch_size:
                self._store_batch(batch_embeddings, batch_documents, batch_ids)
                processed += len(batch_embeddings)
                batch_embeddings, batch_documents, batch_ids = [], [], []

        # Store remaining items
        if batch_embeddings:
            self._store_batch(batch_embeddings, batch_documents, batch_ids)
            processed += len(batch_embeddings)

        logger.info(f"Successfully seeded {processed} chunks into ChromaDB")

    def _store_batch(self, embeddings: List, documents: List, ids: List):
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids,
            )
        except Exception as e:
            logger.error(f"Failed to store batch: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Seed ChromaDB with Minecraft action sequence embeddings")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=".data/chunked_dataset_with_embeddings",
    )
    parser.add_argument(
        "--chroma-host",
        type=str,
        default="localhost",
        help="ChromaDB host"
    )
    parser.add_argument(
        "--chroma-port",
        type=int,
        default=8000,
        help="ChromaDB port"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="episodic_memory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for seeding"
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Reset/recreate the collection"
    )

    args = parser.parse_args()

    # Initialize seeder
    seeder = ChromaDBSeeder(args.chroma_host, args.chroma_port)

    # Connect to ChromaDB
    if not seeder.connect():
        logger.error("Failed to connect to ChromaDB")
        sys.exit(1)

    # Create collection
    seeder.create_collection(args.collection_name, reset=args.reset_collection)

    # Seed from dataset
    dataset_path = Path(args.dataset_dir)
    seeder.seed_from_dataset(dataset_path, args.batch_size)

if __name__ == "__main__":
    main()
