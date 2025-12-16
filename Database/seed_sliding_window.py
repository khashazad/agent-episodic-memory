#!/usr/bin/env python3
"""
Import sliding window data from CSV to ChromaDB.

This script reads a CSV file created by export_to_csv.py and bulk inserts
the embeddings and descriptions into a ChromaDB collection.
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

# Import ChromaDB
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


class CSVToChromaImporter:
    """Imports CSV data into ChromaDB collection."""

    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8000):
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.client = None
        self.collection = None

    def connect(self) -> bool:
        """Connect to ChromaDB server."""
        try:
            self.client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=Settings(allow_reset=True)
            )

            # Test connection
            heartbeat = self.client.heartbeat()
            logger.info(f"Connected to ChromaDB at {self.chroma_host}:{self.chroma_port}")
            logger.debug(f"Heartbeat: {heartbeat}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            return False

    def create_or_get_collection(self, collection_name: str = "episodic_memory",
                                reset: bool = False) -> bool:
        """Create or get the target collection."""
        try:
            # Check if collection exists
            existing_collections = [c.name for c in self.client.list_collections()]

            if reset and collection_name in existing_collections:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=None  # We provide embeddings directly
            )

            logger.info(f"Using collection: {collection_name}")

            # Get collection info
            count = self.collection.count()
            logger.info(f"Current collection size: {count} items")

            return True

        except Exception as e:
            logger.error(f"Failed to create/get collection: {e}")
            return False

    def parse_embedding(self, embedding_str: str) -> Optional[np.ndarray]:
        """Parse embedding string back to numpy array."""
        try:
            # Parse JSON string to list
            embedding_list = json.loads(embedding_str)
            # Convert to numpy array
            embedding = np.array(embedding_list, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Error parsing embedding: {e}")
            return None

    def validate_csv_row(self, row: Dict) -> Tuple[bool, str]:
        """Validate a CSV row for required fields."""
        required_fields = ['id', 'embedding', 'description']

        for field in required_fields:
            if field not in row or not row[field]:
                return False, f"Missing or empty field: {field}"

        # Test embedding parsing
        embedding = self.parse_embedding(row['embedding'])
        if embedding is None:
            return False, "Invalid embedding format"

        return True, "Valid"

    def read_csv_batch(self, csv_path: Path, batch_size: int = 100,
                      start_row: int = 0) -> Tuple[List[Dict], bool]:
        """Read a batch of rows from CSV file."""
        batch_data = []
        has_more = False

        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                # Skip to start position
                for _ in range(start_row):
                    try:
                        next(reader)
                    except StopIteration:
                        break

                # Read batch
                for i, row in enumerate(reader):
                    if i >= batch_size:
                        has_more = True
                        break
                    batch_data.append(row)

                # Check if there are more rows
                if not has_more:
                    try:
                        next(reader)
                        has_more = True
                    except StopIteration:
                        has_more = False

        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return [], False

        return batch_data, has_more

    def prepare_batch_for_chromadb(self, csv_batch: List[Dict]) -> Tuple[List, List, List]:
        """Prepare batch data for ChromaDB insertion."""
        embeddings = []
        documents = []
        ids = []

        for row in csv_batch:
            # Parse embedding
            embedding = self.parse_embedding(row['embedding'])
            if embedding is None:
                logger.warning(f"Skipping row {row['id']} due to invalid embedding")
                continue

            # Prepare document (description)
            description = row['description'].strip()
            if not description:
                logger.warning(f"Skipping row {row['id']} due to empty description")
                continue

            # Add to batch
            embeddings.append(embedding.tolist())  # ChromaDB expects lists
            documents.append(description)
            ids.append(row['id'])

        return embeddings, documents, ids

    def insert_batch(self, embeddings: List, documents: List, ids: List) -> bool:
        """Insert a batch into ChromaDB."""
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids
            )
            return True
        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
            return False

    def import_from_csv(self, csv_path: Path, batch_size: int = 100,
                       dry_run: bool = False, resume_from: int = 0) -> Tuple[int, int]:
        """Import all data from CSV to ChromaDB."""
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 0, 0

        logger.info(f"Starting import from {csv_path}")

        # Count total rows first
        total_rows = 0
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                total_rows = sum(1 for row in reader)
        except Exception as e:
            logger.error(f"Error counting CSV rows: {e}")
            return 0, 0

        logger.info(f"Total rows in CSV: {total_rows}")

        if dry_run:
            logger.info("Dry run mode - validating data without inserting")
            # Validate first few batches
            valid_count = 0
            error_count = 0

            for batch_start in range(0, min(1000, total_rows), batch_size):  # Check first 1000 rows
                csv_batch, _ = self.read_csv_batch(csv_path, batch_size, batch_start)

                for row in csv_batch:
                    is_valid, message = self.validate_csv_row(row)
                    if is_valid:
                        valid_count += 1
                    else:
                        error_count += 1
                        logger.warning(f"Invalid row {row.get('id', 'unknown')}: {message}")

            logger.info(f"Dry run complete: {valid_count} valid, {error_count} invalid")
            return valid_count, valid_count + error_count

        # Real import
        processed_count = 0
        error_count = 0
        current_row = resume_from

        start_time = time.time()

        with tqdm(total=total_rows, initial=resume_from, desc="Importing to ChromaDB") as pbar:
            while current_row < total_rows:
                # Read batch
                csv_batch, has_more = self.read_csv_batch(csv_path, batch_size, current_row)

                if not csv_batch:
                    break

                # Prepare batch for ChromaDB
                embeddings, documents, ids = self.prepare_batch_for_chromadb(csv_batch)

                if not embeddings:
                    logger.warning(f"No valid data in batch starting at row {current_row}")
                    current_row += len(csv_batch)
                    pbar.update(len(csv_batch))
                    continue

                # Insert batch
                success = self.insert_batch(embeddings, documents, ids)

                if success:
                    processed_count += len(embeddings)
                    logger.debug(f"Inserted batch: {len(embeddings)} items")
                else:
                    error_count += len(csv_batch)
                    logger.error(f"Failed to insert batch starting at row {current_row}")

                current_row += len(csv_batch)
                pbar.update(len(csv_batch))

                # Rate limiting to avoid overwhelming the server
                if processed_count % (batch_size * 10) == 0:
                    time.sleep(0.1)

        # Final statistics
        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Import complete:")
        logger.info(f"  Processed: {processed_count} items")
        logger.info(f"  Errors: {error_count} items")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Rate: {processed_count/duration:.2f} items/second")

        # Final collection count
        if self.collection:
            final_count = self.collection.count()
            logger.info(f"Final collection size: {final_count} items")

        return processed_count, total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Import sliding window embeddings from CSV to ChromaDB"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=".data/chroma-data.csv",
        help="Path to CSV file to import"
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
        "--collection-name",
        type=str,
        default=os.getenv("CHROMA_COLLECTION", "episodic_memory"),
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of items to insert in each batch"
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Reset/recreate the collection before import"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without inserting into ChromaDB"
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume import from specific row number"
    )

    args = parser.parse_args()

    # Validate CSV file
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    # Initialize importer
    importer = CSVToChromaImporter(args.chroma_host, args.chroma_port)

    # Connect to ChromaDB
    if not importer.connect():
        logger.error("Failed to connect to ChromaDB")
        sys.exit(1)

    # Create/get collection (skip for dry run)
    if not args.dry_run:
        if not importer.create_or_get_collection(args.collection_name, args.reset_collection):
            logger.error("Failed to create/get collection")
            sys.exit(1)

    # Run import
    try:
        processed, total = importer.import_from_csv(
            csv_path=csv_path,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            resume_from=args.resume_from
        )

        if args.dry_run:
            logger.info("Dry run completed successfully")
        else:
            success_rate = (processed / total * 100) if total > 0 else 0
            logger.info(f"Import success rate: {success_rate:.2f}%")

    except KeyboardInterrupt:
        logger.info("Import interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
