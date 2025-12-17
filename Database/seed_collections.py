#!/usr/bin/env python3
"""
Seed multiple collections in ChromaDB from dataset directory.

This script reads chunk data from the dataset directory and inserts:
1. Collection with video embeddings and action descriptions
2. Collection with video embeddings and LLM-derived descriptions
3. Collection with video embeddings, LLM-derived descriptions, and metadata (chroma-data-v2)

It can also seed from CSV files for backward compatibility.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Marker file to indicate seeding has been completed
SEED_MARKER_FILE = "/chroma/chroma/.seed_complete"


class MultiCollectionSeeder:
    """Seeds multiple ChromaDB collections from dataset directory or CSV files."""

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

    def create_collection(self, collection_name: str, reset: bool = False) -> Optional[object]:
        """Create or get a ChromaDB collection."""
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

    def seed_collection_from_csv(
        self,
        csv_path: Path,
        collection_name: str,
        batch_size: int = 100,
        reset: bool = False
    ) -> Tuple[int, int]:
        """Seed a single collection from a CSV file."""
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
                    # Parse embedding
                    embedding = self.parse_embedding(row['embedding'])
                    if embedding is None:
                        error_count += 1
                        continue

                    # Get description
                    description = row.get('description', '').strip()
                    if not description:
                        error_count += 1
                        continue

                    # Get ID
                    item_id = row.get('id', f"item_{processed_count}")

                    # Extract metadata fields
                    # Note: ChromaDB only accepts str, int, float, bool, SparseVector, or None
                    # Lists and dicts must be JSON-encoded as strings
                    metadata = {}
                    if row.get('episode'):
                        metadata['episode'] = row['episode']
                    if row.get('chunk_index'):
                        try:
                            metadata['chunk_index'] = int(row['chunk_index'])
                        except (ValueError, TypeError):
                            metadata['chunk_index'] = row['chunk_index']
                    if row.get('frame_range'):
                        # Keep as string - don't parse JSON since ChromaDB requires strings
                        # If it's already a list/dict (shouldn't happen from CSV), convert to JSON string
                        frame_range_val = row['frame_range']
                        if isinstance(frame_range_val, (list, dict)):
                            metadata['frame_range'] = json.dumps(frame_range_val)
                        else:
                            # CSV always gives strings, so keep as string
                            metadata['frame_range'] = str(frame_range_val)
                    if row.get('action_summary'):
                        # Keep as string - don't parse JSON since ChromaDB requires strings
                        # If it's already a list/dict (shouldn't happen from CSV), convert to JSON string
                        action_summary_val = row['action_summary']
                        if isinstance(action_summary_val, (list, dict)):
                            metadata['action_summary'] = json.dumps(action_summary_val)
                        else:
                            # CSV always gives strings, so keep as string
                            metadata['action_summary'] = str(action_summary_val)
                    if row.get('action_descriptions'):
                        metadata['action_descriptions'] = row['action_descriptions']
                    if row.get('metadata_path'):
                        metadata['metadata_path'] = row['metadata_path']

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

    def seed_all_collections(
        self,
        csv_collection_pairs: List[Tuple[Path, str]],
        batch_size: int = 100,
        reset: bool = False
    ) -> Dict[str, Tuple[int, int]]:
        """Seed all collections from their respective CSV files."""
        results = {}

        for csv_path, collection_name in csv_collection_pairs:
            processed, total = self.seed_collection_from_csv(
                csv_path=csv_path,
                collection_name=collection_name,
                batch_size=batch_size,
                reset=reset
            )
            results[collection_name] = (processed, total)

        return results

    def _find_llm_description_file(self, chunk_dir: Path) -> Optional[Path]:
        """Find the LLM description file in the chunk directory."""
        possible_names = [
            "llm_derived_description.txt",
            "llm_description.txt",
            "description.txt",
        ]

        for name in possible_names:
            path = chunk_dir / name
            if path.exists():
                return path

        return None

    def get_chunk_dirs(self, dataset_dir: Path) -> List[Path]:
        """Get all chunk directories from the dataset."""
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

    def load_chunk_data(self, chunk_dir: Path) -> Optional[Dict]:
        """Load all data for a single chunk."""
        try:
            # Load embedding
            embedding_path = chunk_dir / "embedding.npy"
            if not embedding_path.exists():
                logger.warning(f"No embedding found in {chunk_dir}")
                return None

            embedding = np.load(embedding_path)

            # Load action descriptions
            actions_path = chunk_dir / "actions.txt"
            action_descriptions = []
            if actions_path.exists():
                with open(actions_path, 'r') as f:
                    action_lines = f.read().strip().split('\n')

                # Clean up action descriptions
                for line in action_lines:
                    if line.strip() and line.startswith('Step'):
                        # Extract description after "Step X: "
                        desc = line.split(': ', 1)[1] if ': ' in line else line
                        action_descriptions.append(desc)

            # Load LLM-derived description
            llm_description = None
            llm_desc_path = self._find_llm_description_file(chunk_dir)
            if llm_desc_path:
                with open(llm_desc_path, 'r', encoding='utf-8') as f:
                    llm_description = f.read().strip()

            # Load metadata if available
            metadata = {}
            metadata_path = chunk_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            return {
                'embedding': embedding,
                'action_descriptions': action_descriptions,
                'llm_description': llm_description,
                'metadata': metadata,
            }

        except Exception as e:
            logger.error(f"Error loading chunk data from {chunk_dir}: {e}")
            return None

    def seed_from_dataset(
        self,
        dataset_dir: Path,
        collection_names: Tuple[str, str, str],
        batch_size: int = 100,
        reset: bool = False
    ) -> Dict[str, Tuple[int, int]]:
        """Seed 3 collections from the dataset directory.

        Args:
            dataset_dir: Path to dataset directory containing episode/chunk structure
            collection_names: Tuple of (collection_1_name, collection_2_name, collection_3_name)
            batch_size: Batch size for seeding
            reset: Whether to reset collections before seeding

        Returns:
            Dictionary mapping collection names to (processed_count, total_chunks) tuples
        """
        chunk_dirs = self.get_chunk_dirs(dataset_dir)

        if not chunk_dirs:
            logger.error(f"No chunks found in {dataset_dir}")
            return {}

        logger.info(f"Found {len(chunk_dirs)} chunks to process")

        collection_1_name, collection_2_name, collection_3_name = collection_names

        # Create all 3 collections
        collection_1 = self.create_collection(collection_1_name, reset=reset)
        collection_2 = self.create_collection(collection_2_name, reset=reset)
        collection_3 = self.create_collection(collection_3_name, reset=reset)

        if collection_1 is None or collection_2 is None or collection_3 is None:
            logger.error("Failed to create one or more collections")
            return {}

        # Check if collections already have data and skip if not resetting
        if not reset:
            if collection_1.count() > 0:
                logger.info(f"Collection '{collection_1_name}' already has {collection_1.count()} items, skipping...")
            if collection_2.count() > 0:
                logger.info(f"Collection '{collection_2_name}' already has {collection_2.count()} items, skipping...")
            if collection_3.count() > 0:
                logger.info(f"Collection '{collection_3_name}' already has {collection_3.count()} items, skipping...")

        # Prepare batches for all three collections
        batch_embeddings_1 = []
        batch_documents_1 = []
        batch_ids_1 = []

        batch_embeddings_2 = []
        batch_documents_2 = []
        batch_ids_2 = []

        batch_embeddings_3 = []
        batch_documents_3 = []
        batch_ids_3 = []
        batch_metadatas_3 = []

        processed_1 = 0
        processed_2 = 0
        processed_3 = 0
        error_count = 0

        for chunk_dir in tqdm(chunk_dirs, desc="Seeding ChromaDB"):
            chunk_data = self.load_chunk_data(chunk_dir)
            if chunk_data is None:
                error_count += 1
                continue

            embedding = chunk_data['embedding'].tolist()

            # Create unique ID from directory path
            episode_name = chunk_dir.parent.name
            chunk_name = chunk_dir.name
            chunk_id = f"{episode_name}_{chunk_name}"

            # Collection 1: Action descriptions
            if chunk_data['action_descriptions']:
                action_doc = "\n".join(chunk_data['action_descriptions'])
                batch_embeddings_1.append(embedding)
                batch_documents_1.append(action_doc)
                batch_ids_1.append(chunk_id)

                if len(batch_embeddings_1) >= batch_size:
                    try:
                        collection_1.add(
                            embeddings=batch_embeddings_1,
                            documents=batch_documents_1,
                            ids=batch_ids_1
                        )
                        processed_1 += len(batch_embeddings_1)
                        batch_embeddings_1, batch_documents_1, batch_ids_1 = [], [], []
                    except Exception as e:
                        logger.error(f"Failed to insert batch into collection 1: {e}")
                        error_count += len(batch_embeddings_1)
                        batch_embeddings_1, batch_documents_1, batch_ids_1 = [], [], []

            # Collection 2: LLM-derived descriptions
            if chunk_data['llm_description']:
                batch_embeddings_2.append(embedding)
                batch_documents_2.append(chunk_data['llm_description'])
                batch_ids_2.append(chunk_id)

                if len(batch_embeddings_2) >= batch_size:
                    try:
                        collection_2.add(
                            embeddings=batch_embeddings_2,
                            documents=batch_documents_2,
                            ids=batch_ids_2
                        )
                        processed_2 += len(batch_embeddings_2)
                        batch_embeddings_2, batch_documents_2, batch_ids_2 = [], [], []
                    except Exception as e:
                        logger.error(f"Failed to insert batch into collection 2: {e}")
                        error_count += len(batch_embeddings_2)
                        batch_embeddings_2, batch_documents_2, batch_ids_2 = [], [], []

            # Collection 3: chroma-data-v2 (LLM descriptions with metadata)
            if chunk_data['llm_description']:
                # Prepare metadata for chroma-data-v2
                metadata = {}
                if chunk_data['metadata']:
                    # Extract relevant metadata fields
                    if 'episode' in chunk_data['metadata']:
                        metadata['episode'] = chunk_data['metadata']['episode']
                    if 'chunk_index' in chunk_data['metadata']:
                        metadata['chunk_index'] = chunk_data['metadata']['chunk_index']
                    if 'frame_range' in chunk_data['metadata']:
                        metadata['frame_range'] = json.dumps(chunk_data['metadata']['frame_range'])
                    if 'action_summary' in chunk_data['metadata']:
                        metadata['action_summary'] = json.dumps(chunk_data['metadata']['action_summary'])
                    if chunk_data['action_descriptions']:
                        metadata['action_descriptions'] = "\n".join(chunk_data['action_descriptions'])

                batch_embeddings_3.append(embedding)
                batch_documents_3.append(chunk_data['llm_description'])
                batch_ids_3.append(chunk_id)
                batch_metadatas_3.append(metadata if metadata else None)

                if len(batch_embeddings_3) >= batch_size:
                    try:
                        collection_3.add(
                            embeddings=batch_embeddings_3,
                            documents=batch_documents_3,
                            ids=batch_ids_3,
                            metadatas=batch_metadatas_3
                        )
                        processed_3 += len(batch_embeddings_3)
                        batch_embeddings_3, batch_documents_3, batch_ids_3, batch_metadatas_3 = [], [], [], []
                    except Exception as e:
                        logger.error(f"Failed to insert batch into collection 3: {e}")
                        error_count += len(batch_embeddings_3)
                        batch_embeddings_3, batch_documents_3, batch_ids_3, batch_metadatas_3 = [], [], [], []

        # Store remaining items for all collections
        if batch_embeddings_1:
            try:
                collection_1.add(
                    embeddings=batch_embeddings_1,
                    documents=batch_documents_1,
                    ids=batch_ids_1
                )
                processed_1 += len(batch_embeddings_1)
            except Exception as e:
                logger.error(f"Failed to insert final batch into collection 1: {e}")
                error_count += len(batch_embeddings_1)

        if batch_embeddings_2:
            try:
                collection_2.add(
                    embeddings=batch_embeddings_2,
                    documents=batch_documents_2,
                    ids=batch_ids_2
                )
                processed_2 += len(batch_embeddings_2)
            except Exception as e:
                logger.error(f"Failed to insert final batch into collection 2: {e}")
                error_count += len(batch_embeddings_2)

        if batch_embeddings_3:
            try:
                collection_3.add(
                    embeddings=batch_embeddings_3,
                    documents=batch_documents_3,
                    ids=batch_ids_3,
                    metadatas=batch_metadatas_3
                )
                processed_3 += len(batch_embeddings_3)
            except Exception as e:
                logger.error(f"Failed to insert final batch into collection 3: {e}")
                error_count += len(batch_embeddings_3)

        total_chunks = len(chunk_dirs)
        results = {
            collection_1_name: (processed_1, total_chunks),
            collection_2_name: (processed_2, total_chunks),
            collection_3_name: (processed_3, total_chunks),
        }

        logger.info(f"Seeding completed: {processed_1} items in collection 1, {processed_2} in collection 2, {processed_3} in collection 3, {error_count} errors")

        return results


def check_seed_marker() -> bool:
    """Check if seeding has already been completed."""
    return os.path.exists(SEED_MARKER_FILE)


def create_seed_marker():
    """Create marker file to indicate seeding is complete."""
    try:
        marker_dir = os.path.dirname(SEED_MARKER_FILE)
        os.makedirs(marker_dir, exist_ok=True)
        with open(SEED_MARKER_FILE, 'w') as f:
            f.write(f"Seeded at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logger.info(f"Created seed marker: {SEED_MARKER_FILE}")
    except Exception as e:
        logger.warning(f"Could not create seed marker: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Seed multiple ChromaDB collections from dataset directory or CSV files"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory (takes precedence over CSV files)"
    )
    parser.add_argument(
        "--collection-1",
        type=str,
        default="episodic_memory_v1",
        help="Collection name for action descriptions (dataset mode) or first CSV collection (CSV mode)"
    )
    parser.add_argument(
        "--collection-2",
        type=str,
        default="episodic_memory_v2",
        help="Collection name for LLM-derived descriptions (dataset mode) or second CSV collection (CSV mode)"
    )
    parser.add_argument(
        "--collection-3",
        type=str,
        default="episodic_memory_v3",
        help="Collection name for chroma-data-v2 (LLM descriptions with metadata) - dataset mode only"
    )
    parser.add_argument(
        "--csv-v1",
        type=str,
        default=".data/chroma-data-v1.csv",
        help="Path to first CSV file (chroma-data-v1.csv) - used if --dataset-dir not provided"
    )
    parser.add_argument(
        "--csv-v2",
        type=str,
        default=".data/chroma-data-v2.csv",
        help="Path to second CSV file (chroma-data-v2.csv) - used if --dataset-dir not provided"
    )
    parser.add_argument(
        "--chroma-host",
        type=str,
        default=os.getenv("CHROMA_HOST", "0.0.0.0"),
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
        default=5000,
        help="Number of items to insert per batch"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset/recreate collections before seeding"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force seeding even if already completed"
    )
    parser.add_argument(
        "--skip-marker",
        action="store_true",
        help="Skip checking/creating seed marker file"
    )

    args = parser.parse_args()

    # Check if already seeded (unless forced or skipped)
    if not args.skip_marker and not args.force and check_seed_marker():
        logger.info("Database already seeded (marker file exists). Use --force to re-seed.")
        sys.exit(0)

    # Initialize seeder
    seeder = MultiCollectionSeeder(args.chroma_host, args.chroma_port)

    # Wait for ChromaDB to be available
    if not seeder.wait_for_chromadb():
        logger.error("ChromaDB is not available")
        sys.exit(1)

    try:
        # Seed from dataset directory if provided, otherwise use CSV files
        if args.dataset_dir:
            dataset_path = Path(args.dataset_dir)
            if not dataset_path.exists():
                logger.error(f"Dataset directory not found: {dataset_path}")
                sys.exit(1)

            logger.info("Seeding from dataset directory...")
            logger.info(f"  Collection 1 ({args.collection_1}): Action descriptions")
            logger.info(f"  Collection 2 ({args.collection_2}): LLM-derived descriptions")
            logger.info(f"  Collection 3 ({args.collection_3}): chroma-data-v2")

            try:
                results = seeder.seed_from_dataset(
                    dataset_dir=dataset_path,
                    collection_names=(args.collection_1, args.collection_2, args.collection_3),
                    batch_size=args.batch_size,
                    reset=args.reset
                )
            except Exception as e:
                logger.error(f"Seeding from dataset failed: {e}")
                sys.exit(1)
        else:
            # Fall back to CSV seeding for backward compatibility
            logger.info("Seeding from CSV files...")
            csv_collection_pairs = []

            csv_v1 = Path(args.csv_v1)
            csv_v2 = Path(args.csv_v2)

            if csv_v1.exists():
                csv_collection_pairs.append((csv_v1, args.collection_1))
            else:
                logger.warning(f"CSV file not found: {csv_v1}")

            if csv_v2.exists():
                csv_collection_pairs.append((csv_v2, args.collection_2))
            else:
                logger.warning(f"CSV file not found: {csv_v2}")

            if not csv_collection_pairs:
                logger.error("No valid CSV files found and --dataset-dir not provided")
                sys.exit(1)

            try:
                results = seeder.seed_all_collections(
                    csv_collection_pairs=csv_collection_pairs,
                    batch_size=args.batch_size,
                    reset=args.reset
                )
            except Exception as e:
                logger.error(f"Seeding from CSV files failed: {e}")
                sys.exit(1)

        # Print summary
        logger.info("=" * 50)
        logger.info("SEEDING SUMMARY")
        logger.info("=" * 50)

        total_processed = 0
        total_items = 0

        for collection_name, (processed, total) in results.items():
            success_rate = (processed / total * 100) if total > 0 else 0
            logger.info(f"  {collection_name}: {processed}/{total} ({success_rate:.1f}%)")
            total_processed += processed
            total_items += total

        logger.info(f"  TOTAL: {total_processed}/{total_items}")
        logger.info("=" * 50)

        # Create seed marker if successful
        if not args.skip_marker and total_processed > 0:
            create_seed_marker()

        logger.info("Seeding completed successfully!")

    except KeyboardInterrupt:
        logger.info("Seeding interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
