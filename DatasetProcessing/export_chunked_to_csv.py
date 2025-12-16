#!/usr/bin/env python3
"""
Export chunked dataset with embeddings and LLM descriptions to CSV format for ChromaDB insertion.

This script scans the chunked_dataset_with_embeddings_and_llm_descriptions dataset and extracts:
- Video embeddings (MineCLIP)
- LLM-derived descriptions
- Action summaries
- Metadata (episode, chunk index, frame range)

Output CSV format:
id,embedding,description,episode,chunk_index,frame_range,action_summary,metadata_path
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChunkedDatasetCSVExporter:
    """Exports chunked dataset with embeddings and LLM descriptions to CSV format."""

    def __init__(self, dataset_dir: Path, output_csv: Path, resume: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.output_csv = Path(output_csv)
        self.resume = resume
        self.processed_ids = set()

        # Load existing IDs if resuming
        if self.resume and self.output_csv.exists():
            self._load_existing_ids()

    def _load_existing_ids(self):
        """Load IDs from existing CSV file to support resume functionality."""
        try:
            with open(self.output_csv, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.processed_ids.add(row['id'])
            logger.info(f"Loaded {len(self.processed_ids)} existing IDs from {self.output_csv}")
        except Exception as e:
            logger.error(f"Error loading existing CSV: {e}")

    def get_chunk_dirs(self) -> List[Path]:
        """Get all chunk directories from the dataset."""
        chunk_dirs = []

        if not self.dataset_dir.exists():
            logger.error(f"Dataset directory not found: {self.dataset_dir}")
            return chunk_dirs

        for episode_dir in self.dataset_dir.iterdir():
            if episode_dir.is_dir():
                for chunk_dir in episode_dir.iterdir():
                    if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk_"):
                        chunk_dirs.append(chunk_dir)

        return sorted(chunk_dirs)

    def _find_description_file(self, chunk_dir: Path) -> Optional[Path]:
        """Find the LLM description file in the chunk directory.

        Checks for common naming patterns:
        - llm_derived_description.txt
        - llm_description.txt
        - description.txt
        """
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

    def load_chunk_data(self, chunk_dir: Path) -> Optional[Dict]:
        """Load all data for a single chunk."""
        try:
            # Create chunk ID
            episode_name = chunk_dir.parent.name
            chunk_name = chunk_dir.name
            chunk_id = f"{episode_name}_{chunk_name}"

            # Skip if already processed (resume mode)
            if chunk_id in self.processed_ids:
                return None

            # Load embedding
            embedding_path = chunk_dir / "embedding.npy"
            if not embedding_path.exists():
                logger.warning(f"No embedding found in {chunk_dir}")
                return None

            embedding = np.load(embedding_path)

            # Load LLM description
            description_path = self._find_description_file(chunk_dir)
            if description_path is None:
                logger.warning(f"No LLM description found in {chunk_dir}")
                return None

            with open(description_path, 'r', encoding='utf-8') as f:
                description = f.read().strip()

            if not description:
                logger.warning(f"Empty description in {chunk_dir}")
                return None

            # Load metadata
            metadata_path = chunk_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading metadata from {metadata_path}: {e}")

            # Extract chunk index and frame range
            chunk_index = None
            frame_range = None
            action_summary = None

            if metadata:
                chunk_index = metadata.get('chunk_index')
                frame_range = metadata.get('frame_range')
                action_summary = metadata.get('action_summary')

            # Fallback: extract chunk index from directory name
            if chunk_index is None:
                try:
                    chunk_index = int(chunk_name.split("_")[1])
                except (ValueError, IndexError):
                    logger.warning(f"Could not extract chunk index from {chunk_name}")

            # Load action descriptions as fallback/supplement
            actions_path = chunk_dir / "actions.txt"
            action_descriptions = None
            if actions_path.exists():
                try:
                    with open(actions_path, 'r', encoding='utf-8') as f:
                        action_descriptions = f.read().strip()
                except Exception as e:
                    logger.warning(f"Error loading actions from {actions_path}: {e}")

            return {
                'id': chunk_id,
                'embedding': embedding,
                'description': description,
                'episode': episode_name,
                'chunk_index': chunk_index,
                'frame_range': frame_range,
                'action_summary': action_summary,
                'action_descriptions': action_descriptions,
                'metadata_path': str(metadata_path) if metadata_path.exists() else None
            }

        except Exception as e:
            logger.error(f"Error loading chunk data from {chunk_dir}: {e}")
            return None

    def embedding_to_string(self, embedding: np.ndarray) -> str:
        """Convert embedding array to string format for CSV."""
        # Convert to list and then to compact JSON string
        return json.dumps(embedding.tolist())

    def export_to_csv(self, dry_run: bool = False, batch_size: int = 1000) -> Tuple[int, int]:
        """Export all chunks to CSV file."""
        chunk_dirs = self.get_chunk_dirs()

        if not chunk_dirs:
            logger.error(f"No chunks found in {self.dataset_dir}")
            return 0, 0

        logger.info(f"Found {len(chunk_dirs)} chunks to process")

        if dry_run:
            logger.info("Dry run mode - no CSV file will be written")
            # Just validate data loading
            valid_count = 0
            for chunk_dir in tqdm(chunk_dirs[:10], desc="Dry run validation"):  # Check first 10
                chunk_data = self.load_chunk_data(chunk_dir)
                if chunk_data:
                    valid_count += 1
                    logger.debug(f"Valid: {chunk_data['id']} - {len(chunk_data['description'])} chars")
            logger.info(f"Dry run: {valid_count}/10 chunks have valid data")
            return valid_count, 10

        # Open CSV file for writing
        mode = 'a' if self.resume and self.output_csv.exists() else 'w'
        write_header = mode == 'w' or not self.output_csv.exists()

        processed_count = 0
        skipped_count = 0
        error_count = 0

        with open(self.output_csv, mode, newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'id', 'embedding', 'description', 'episode',
                'chunk_index', 'frame_range', 'action_summary',
                'action_descriptions', 'metadata_path'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            batch_data = []

            for chunk_dir in tqdm(chunk_dirs, desc="Exporting chunks"):
                try:
                    chunk_data = self.load_chunk_data(chunk_dir)

                    if chunk_data is None:
                        skipped_count += 1
                        continue

                    # Prepare CSV row
                    csv_row = {
                        'id': chunk_data['id'],
                        'embedding': self.embedding_to_string(chunk_data['embedding']),
                        'description': chunk_data['description'],
                        'episode': chunk_data['episode'],
                        'chunk_index': chunk_data['chunk_index'],
                        'frame_range': json.dumps(chunk_data['frame_range']) if chunk_data['frame_range'] else None,
                        'action_summary': json.dumps(chunk_data['action_summary']) if chunk_data['action_summary'] else None,
                        'action_descriptions': chunk_data['action_descriptions'],
                        'metadata_path': chunk_data['metadata_path']
                    }

                    batch_data.append(csv_row)

                    # Write batch when full
                    if len(batch_data) >= batch_size:
                        writer.writerows(batch_data)
                        csvfile.flush()  # Ensure data is written
                        processed_count += len(batch_data)
                        batch_data = []

                except Exception as e:
                    logger.error(f"Error processing {chunk_dir}: {e}")
                    error_count += 1
                    continue

            # Write remaining data
            if batch_data:
                writer.writerows(batch_data)
                processed_count += len(batch_data)

        logger.info(f"Export complete:")
        logger.info(f"  Processed: {processed_count} chunks")
        logger.info(f"  Skipped: {skipped_count} chunks")
        logger.info(f"  Errors: {error_count} chunks")
        logger.info(f"  Output CSV: {self.output_csv}")

        return processed_count, len(chunk_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Export chunked dataset with LLM descriptions to CSV format"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=".data/chunked_dataset_with_embeddings_and_llm_descriptions",
        help="Path to chunked dataset directory with embeddings and LLM descriptions"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=".data/chroma-data-v1.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume export by appending to existing CSV file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without writing CSV file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of rows to write at once"
    )

    args = parser.parse_args()

    # Validate input directory
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Create output directory if needed
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Initialize exporter
    exporter = ChunkedDatasetCSVExporter(
        dataset_dir=dataset_dir,
        output_csv=output_csv,
        resume=args.resume
    )

    # Run export
    try:
        processed, total = exporter.export_to_csv(
            dry_run=args.dry_run,
            batch_size=args.batch_size
        )

        if not args.dry_run:
            # Print final statistics
            file_size = output_csv.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"CSV file size: {file_size:.2f} MB")

            # Validate CSV structure
            try:
                with open(output_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    row_count = sum(1 for row in reader)
                logger.info(f"CSV validation: {row_count} rows written")
            except Exception as e:
                logger.warning(f"CSV validation failed: {e}")

    except KeyboardInterrupt:
        logger.info("Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
