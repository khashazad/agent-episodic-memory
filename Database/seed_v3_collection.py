#!/usr/bin/env python3
"""
Seed episodic_memory_v3 collection in ChromaDB from vector_db.csv.

This script reads the fused embeddings from the vector_db.csv file and
inserts them into a ChromaDB collection named "episodic_memory_v3".

The CSV contains:
- id: Unique identifier
- fused_embedding: 512-dim fused embedding (video + text average)
- video_embedding: 512-dim video embedding
- text_embedding: 512-dim text embedding
- next_action: JSON action dict for the next frame
- description: LLM-generated description
- episode: Episode name
- window_index: Window index within episode
- frame_range: [start, end] frame indices
- metadata: Full metadata JSON

Usage:
    python Database/seed_v3_collection.py --csv-path .data/vector_db.csv
    python Database/seed_v3_collection.py --csv-path .data/vector_db.csv --reset
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Error: ChromaDB not installed. Run: pip install chromadb")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class V3CollectionSeeder:
    """Seeds ChromaDB episodic_memory_v3 collection from vector_db.csv."""

    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8000):
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.client = None

    def wait_for_chromadb(self, max_retries: int = 30, retry_delay: int = 2) -> bool:
        """Wait for ChromaDB to become available."""
        logger.info(f"Waiting for ChromaDB at {self.chroma_host}:{self.chroma_port}...")

        for attempt in range(max_retries):
            try:
                self.client = chromadb.HttpClient(
                    host=self.chroma_host,
                    port=self.chroma_port,
                    settings=Settings(allow_reset=True)
                )
                heartbeat = self.client.heartbeat()
                logger.info(f"ChromaDB is ready (heartbeat: {heartbeat})")
                return True
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        logger.error(f"ChromaDB not available after {max_retries} attempts")
        return False

    def create_collection(
        self,
        collection_name: str = "episodic_memory_v3",
        reset: bool = False
    ) -> Optional[object]:
        """Create or get the ChromaDB collection."""
        try:
            existing_collections = [c.name for c in self.client.list_collections()]

            if reset and collection_name in existing_collections:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")

            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=None  # We provide embeddings directly
            )

            logger.info(f"Using collection: {collection_name} (current size: {collection.count()})")
            return collection

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return None

    def parse_embedding(self, embedding_str: str) -> Optional[np.ndarray]:
        """Parse embedding string back to numpy array."""
        try:
            embedding_list = json.loads(embedding_str)
            return np.array(embedding_list, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error parsing embedding: {e}")
            return None

    def count_csv_rows(self, csv_path: Path) -> int:
        """Count total rows in CSV file."""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return sum(1 for _ in reader)
        except Exception as e:
            logger.error(f"Error counting CSV rows: {e}")
            return 0

    def seed_from_csv(
        self,
        csv_path: Path,
        collection_name: str = "episodic_memory_v3",
        batch_size: int = 1000,
        reset: bool = False
    ) -> Tuple[int, int]:
        """Seed the collection from vector_db.csv.

        Args:
            csv_path: Path to vector_db.csv
            collection_name: Name of the ChromaDB collection
            batch_size: Number of items to insert per batch
            reset: Whether to reset the collection before seeding

        Returns:
            Tuple of (processed_count, total_rows)
        """
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 0, 0

        logger.info(f"Seeding collection '{collection_name}' from {csv_path}")

        # Create collection
        collection = self.create_collection(collection_name, reset=reset)
        if collection is None:
            return 0, 0

        # Check if collection already has data and skip if not resetting
        if not reset and collection.count() > 0:
            logger.info(f"Collection '{collection_name}' already has {collection.count()} items, skipping...")
            return collection.count(), collection.count()

        # Count total rows
        total_rows = self.count_csv_rows(csv_path)
        logger.info(f"Total rows in CSV: {total_rows}")

        processed_count = 0
        error_count = 0

        batch_embeddings = []
        batch_documents = []
        batch_ids = []
        batch_metadatas = []

        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in tqdm(reader, total=total_rows, desc=f"Seeding {collection_name}"):
                try:
                    # Parse fused embedding (primary embedding for queries)
                    embedding = self.parse_embedding(row['fused_embedding'])
                    if embedding is None:
                        error_count += 1
                        continue

                    # Get description as document
                    description = row.get('description', '').strip()
                    if not description:
                        description = "No description available"

                    # Get ID
                    item_id = row.get('id', f"item_{processed_count}")

                    # Build metadata
                    metadata = {}

                    # Core fields
                    if row.get('episode'):
                        metadata['episode'] = row['episode']

                    if row.get('window_index'):
                        try:
                            metadata['window_index'] = int(row['window_index'])
                        except (ValueError, TypeError):
                            metadata['window_index'] = str(row['window_index'])

                    if row.get('frame_range'):
                        metadata['frame_range'] = str(row['frame_range'])

                    # Store next_action as JSON string in metadata
                    if row.get('next_action'):
                        metadata['next_action'] = str(row['next_action'])

                    # Optionally store video/text embeddings as references
                    # (Not storing these to save space - they can be recomputed if needed)

                    batch_embeddings.append(embedding.tolist())
                    batch_documents.append(description)
                    batch_ids.append(item_id)
                    batch_metadatas.append(metadata if metadata else None)

                    # Insert batch when full
                    if len(batch_embeddings) >= batch_size:
                        try:
                            collection.add(
                                embeddings=batch_embeddings,
                                documents=batch_documents,
                                ids=batch_ids,
                                metadatas=batch_metadatas
                            )
                            processed_count += len(batch_embeddings)
                            logger.info(f"Inserted batch: {processed_count}/{total_rows}")
                        except Exception as e:
                            logger.error(f"Failed to insert batch: {e}")
                            error_count += len(batch_embeddings)

                        batch_embeddings = []
                        batch_documents = []
                        batch_ids = []
                        batch_metadatas = []

                except Exception as e:
                    logger.error(f"Error processing row: {e}")
                    error_count += 1

        # Insert remaining items
        if batch_embeddings:
            try:
                collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
                processed_count += len(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to insert final batch: {e}")
                error_count += len(batch_embeddings)

        logger.info(f"Collection '{collection_name}': {processed_count} items inserted, {error_count} errors")
        return processed_count, total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Seed episodic_memory_v3 collection from vector_db.csv"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=".data/vector_db.csv",
        help="Path to vector_db.csv file"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="episodic_memory_v3",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--chroma-host",
        type=str,
        default=os.getenv("CHROMA_HOST", "localhost"),
        help="ChromaDB host"
    )
    parser.add_argument(
        "--chroma-port",
        type=int,
        default=int(os.getenv("CHROMA_PORT", "8000")),
        help="ChromaDB port"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of items to insert per batch"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset/recreate collection before seeding"
    )

    args = parser.parse_args()

    # Initialize seeder
    seeder = V3CollectionSeeder(args.chroma_host, args.chroma_port)

    # Wait for ChromaDB
    if not seeder.wait_for_chromadb():
        logger.error("ChromaDB is not available")
        sys.exit(1)

    try:
        csv_path = Path(args.csv_path)

        processed, total = seeder.seed_from_csv(
            csv_path=csv_path,
            collection_name=args.collection_name,
            batch_size=args.batch_size,
            reset=args.reset
        )

        # Print summary
        logger.info("=" * 50)
        logger.info("SEEDING SUMMARY")
        logger.info("=" * 50)

        if total > 0:
            success_rate = (processed / total * 100)
            logger.info(f"  Collection: {args.collection_name}")
            logger.info(f"  Processed: {processed}/{total} ({success_rate:.1f}%)")
        else:
            logger.info("  No data to process")

        logger.info("=" * 50)
        logger.info("Seeding completed successfully!")

    except KeyboardInterrupt:
        logger.info("Seeding interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
