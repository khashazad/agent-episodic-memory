#!/usr/bin/env python3
"""
Export sliding window dataset to CSV format for ChromaDB insertion.

This script scans a sliding window dataset and extracts:
- Video embeddings (MineCLIP)
- LLM-derived descriptions
- Metadata (episode, window index, frame range)

Output CSV format:
id,embedding,description,episode,window_index,frame_range,metadata_path
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


class SlidingWindowCSVExporter:
    """Exports sliding window dataset to CSV format."""

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

    def get_window_dirs(self) -> List[Path]:
        """Get all window directories from the dataset."""
        window_dirs = []

        if not self.dataset_dir.exists():
            logger.error(f"Dataset directory not found: {self.dataset_dir}")
            return window_dirs

        for episode_dir in self.dataset_dir.iterdir():
            if episode_dir.is_dir():
                for window_dir in episode_dir.iterdir():
                    if window_dir.is_dir() and (window_dir.name.startswith("window_") or window_dir.name.startswith("chunk_")):
                        window_dirs.append(window_dir)

        return sorted(window_dirs)

    def load_window_data(self, window_dir: Path) -> Optional[Dict]:
        """Load all data for a single window."""
        try:
            # Create window ID
            episode_name = window_dir.parent.name
            window_name = window_dir.name
            window_id = f"{episode_name}_{window_name}"

            # Skip if already processed (resume mode)
            if window_id in self.processed_ids:
                return None

            # Load embedding
            embedding_path = window_dir / "embedding.npy"
            if not embedding_path.exists():
                logger.warning(f"No embedding found in {window_dir}")
                return None

            embedding = np.load(embedding_path)

            # Load LLM description
            description_path = window_dir / "llm_derived_description.txt"
            if not description_path.exists():
                logger.warning(f"No LLM description found in {window_dir}")
                return None

            with open(description_path, 'r', encoding='utf-8') as f:
                description = f.read().strip()

            if not description:
                logger.warning(f"Empty description in {window_dir}")
                return None

            # Load metadata
            metadata_path = window_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading metadata from {metadata_path}: {e}")

            # Extract window index and frame range
            window_index = None
            frame_range = None

            if metadata:
                window_index = metadata.get('window_index', metadata.get('chunk_index'))
                frame_range = metadata.get('frame_range')

            # Fallback: extract window index from directory name
            if window_index is None:
                try:
                    if window_name.startswith("window_"):
                        window_index = int(window_name.split("_")[1])
                    elif window_name.startswith("chunk_"):
                        window_index = int(window_name.split("_")[1])
                except ValueError:
                    logger.warning(f"Could not extract window index from {window_name}")

            return {
                'id': window_id,
                'embedding': embedding,
                'description': description,
                'episode': episode_name,
                'window_index': window_index,
                'frame_range': frame_range,
                'metadata_path': str(metadata_path) if metadata_path.exists() else None
            }

        except Exception as e:
            logger.error(f"Error loading window data from {window_dir}: {e}")
            return None

    def embedding_to_string(self, embedding: np.ndarray) -> str:
        """Convert embedding array to string format for CSV."""
        # Convert to list and then to compact JSON string
        return json.dumps(embedding.tolist())

    def export_to_csv(self, dry_run: bool = False, batch_size: int = 1000) -> Tuple[int, int]:
        """Export all windows to CSV file."""
        window_dirs = self.get_window_dirs()

        if not window_dirs:
            logger.error(f"No windows found in {self.dataset_dir}")
            return 0, 0

        logger.info(f"Found {len(window_dirs)} windows to process")

        if dry_run:
            logger.info("Dry run mode - no CSV file will be written")
            # Just validate data loading
            valid_count = 0
            for window_dir in tqdm(window_dirs[:10], desc="Dry run validation"):  # Check first 10
                window_data = self.load_window_data(window_dir)
                if window_data:
                    valid_count += 1
            logger.info(f"Dry run: {valid_count}/10 windows have valid data")
            return valid_count, 10

        # Open CSV file for writing
        mode = 'a' if self.resume and self.output_csv.exists() else 'w'
        write_header = mode == 'w' or not self.output_csv.exists()

        processed_count = 0
        skipped_count = 0
        error_count = 0

        with open(self.output_csv, mode, newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'embedding', 'description', 'episode', 'window_index', 'frame_range', 'metadata_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            batch_data = []

            for window_dir in tqdm(window_dirs, desc="Exporting windows"):
                try:
                    window_data = self.load_window_data(window_dir)

                    if window_data is None:
                        skipped_count += 1
                        continue

                    # Prepare CSV row
                    csv_row = {
                        'id': window_data['id'],
                        'embedding': self.embedding_to_string(window_data['embedding']),
                        'description': window_data['description'],
                        'episode': window_data['episode'],
                        'window_index': window_data['window_index'],
                        'frame_range': json.dumps(window_data['frame_range']) if window_data['frame_range'] else None,
                        'metadata_path': window_data['metadata_path']
                    }

                    batch_data.append(csv_row)

                    # Write batch when full
                    if len(batch_data) >= batch_size:
                        writer.writerows(batch_data)
                        csvfile.flush()  # Ensure data is written
                        processed_count += len(batch_data)
                        batch_data = []

                except Exception as e:
                    logger.error(f"Error processing {window_dir}: {e}")
                    error_count += 1
                    continue

            # Write remaining data
            if batch_data:
                writer.writerows(batch_data)
                processed_count += len(batch_data)

        logger.info(f"Export complete:")
        logger.info(f"  Processed: {processed_count} windows")
        logger.info(f"  Skipped: {skipped_count} windows")
        logger.info(f"  Errors: {error_count} windows")
        logger.info(f"  Output CSV: {self.output_csv}")

        return processed_count, len(window_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Export sliding window dataset to CSV format"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=".data/sliding_window_dataset_complete",
        help="Path to sliding window dataset directory"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=".data/chroma-data.csv",
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
        default=1000,
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
    exporter = SlidingWindowCSVExporter(
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
