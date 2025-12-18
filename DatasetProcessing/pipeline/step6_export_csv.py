#!/usr/bin/env python3
"""
Step 6: Export CSV

Creates a CSV file containing fused embeddings (average of video and text)
and LLM-derived actions for seeding the vector database.

Inputs per window:
- video_embedding.npy (512-dim)
- text_embedding.npy (512-dim)
- next_action.json (action dict)
- description.txt (scene description)
- metadata.json (window metadata)

Output:
- vectordb_data.csv with columns:
  id, fused_embedding, video_embedding, text_embedding,
  next_action, description, episode, window_index, metadata
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from . import get_window_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CSVExporter:
    """Exports processed windows to CSV format for vectorDB."""

    def __init__(
        self,
        data_dir: Path,
        output_csv: Path,
        resume: bool = False
    ):
        """Initialize the CSV exporter.

        Args:
            data_dir: Path to dataset with processed windows
            output_csv: Output CSV file path
            resume: If True, append to existing CSV
        """
        self.data_dir = Path(data_dir)
        self.output_csv = Path(output_csv)
        self.resume = resume
        self.processed_ids = set()

        if self.resume and self.output_csv.exists():
            self._load_existing_ids()

    def _load_existing_ids(self):
        """Load IDs from existing CSV to support resume."""
        try:
            with open(self.output_csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.processed_ids.add(row['id'])
            logger.info(f"Loaded {len(self.processed_ids)} existing IDs from {self.output_csv}")
        except Exception as e:
            logger.warning(f"Error loading existing CSV: {e}")

    def compute_fused_embedding(
        self,
        video_embedding: np.ndarray,
        text_embedding: np.ndarray
    ) -> np.ndarray:
        """Compute fused embedding as average of video and text embeddings.

        Args:
            video_embedding: Video embedding [512]
            text_embedding: Text embedding [512]

        Returns:
            Fused embedding [512]
        """
        return (video_embedding + text_embedding) / 2.0

    def load_window_data(self, window_dir: Path) -> Optional[Dict]:
        """Load all required data from a window directory.

        Args:
            window_dir: Path to window directory

        Returns:
            Dict with all window data or None if incomplete
        """
        episode_name = window_dir.parent.name
        window_name = window_dir.name
        window_id = f"{episode_name}_{window_name}"

        # Skip if already processed
        if window_id in self.processed_ids:
            return None

        # Check required files
        video_emb_path = window_dir / "video_embedding.npy"
        text_emb_path = window_dir / "text_embedding.npy"
        action_path = window_dir / "next_action.json"
        description_path = window_dir / "description.txt"
        metadata_path = window_dir / "metadata.json"

        # Load video embedding (required)
        if not video_emb_path.exists():
            logger.debug(f"Missing video_embedding.npy in {window_dir}")
            return None
        video_embedding = np.load(video_emb_path)

        # Load text embedding (required)
        if not text_emb_path.exists():
            logger.debug(f"Missing text_embedding.npy in {window_dir}")
            return None
        text_embedding = np.load(text_emb_path)

        # Load action (required)
        if not action_path.exists():
            logger.debug(f"Missing next_action.json in {window_dir}")
            return None
        with open(action_path, 'r', encoding='utf-8') as f:
            next_action = json.load(f)

        # Load description (required)
        if not description_path.exists():
            logger.debug(f"Missing description.txt in {window_dir}")
            return None
        with open(description_path, 'r', encoding='utf-8') as f:
            description = f.read().strip()

        # Load metadata (optional)
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metadata from {window_dir}: {e}")

        # Compute fused embedding
        fused_embedding = self.compute_fused_embedding(video_embedding, text_embedding)

        return {
            'id': window_id,
            'fused_embedding': fused_embedding,
            'video_embedding': video_embedding,
            'text_embedding': text_embedding,
            'next_action': next_action,
            'description': description,
            'episode': episode_name,
            'window_index': metadata.get('window_index'),
            'frame_range': metadata.get('frame_range'),
            'metadata': metadata
        }

    @staticmethod
    def embedding_to_string(embedding: np.ndarray) -> str:
        """Convert embedding array to JSON string.

        Args:
            embedding: NumPy array

        Returns:
            JSON string representation
        """
        return json.dumps(embedding.tolist())

    def export(self, batch_size: int = 1000) -> Tuple[int, int]:
        """Export all windows to CSV.

        Args:
            batch_size: Number of rows to write at once

        Returns:
            Tuple of (processed count, total windows)
        """
        window_dirs = get_window_dirs(self.data_dir)

        if not window_dirs:
            logger.error(f"No windows found in {self.data_dir}")
            return 0, 0

        logger.info(f"Found {len(window_dirs)} windows to process")

        # Create output directory if needed
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Determine write mode
        mode = 'a' if self.resume and self.output_csv.exists() else 'w'
        write_header = mode == 'w' or not self.output_csv.exists()

        processed_count = 0
        skipped_count = 0

        fieldnames = [
            'id',
            'fused_embedding',
            'video_embedding',
            'text_embedding',
            'next_action',
            'description',
            'episode',
            'window_index',
            'frame_range',
            'metadata'
        ]

        with open(self.output_csv, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            batch_data = []

            for window_dir in tqdm(window_dirs, desc="Exporting to CSV"):
                try:
                    window_data = self.load_window_data(window_dir)

                    if window_data is None:
                        skipped_count += 1
                        continue

                    # Prepare CSV row
                    csv_row = {
                        'id': window_data['id'],
                        'fused_embedding': self.embedding_to_string(window_data['fused_embedding']),
                        'video_embedding': self.embedding_to_string(window_data['video_embedding']),
                        'text_embedding': self.embedding_to_string(window_data['text_embedding']),
                        'next_action': json.dumps(window_data['next_action']),
                        'description': window_data['description'],
                        'episode': window_data['episode'],
                        'window_index': window_data['window_index'],
                        'frame_range': json.dumps(window_data['frame_range']) if window_data['frame_range'] else None,
                        'metadata': json.dumps(window_data['metadata']) if window_data['metadata'] else None
                    }

                    batch_data.append(csv_row)

                    # Write batch when full
                    if len(batch_data) >= batch_size:
                        writer.writerows(batch_data)
                        csvfile.flush()
                        processed_count += len(batch_data)
                        batch_data = []

                except Exception as e:
                    logger.error(f"Error processing {window_dir}: {e}")
                    skipped_count += 1
                    continue

            # Write remaining data
            if batch_data:
                writer.writerows(batch_data)
                processed_count += len(batch_data)

        logger.info(f"Export complete:")
        logger.info(f"  Processed: {processed_count} windows")
        logger.info(f"  Skipped: {skipped_count} windows")
        logger.info(f"  Output CSV: {self.output_csv}")

        return processed_count, len(window_dirs)


def run_step6(
    data_dir: Path,
    output_csv: Path,
    resume: bool = False,
    batch_size: int = 1000
) -> bool:
    """Run Step 6: Export to CSV with fused embeddings.

    Args:
        data_dir: Path to dataset with processed windows
        output_csv: Output CSV file path
        resume: If True, append to existing CSV
        batch_size: Batch size for writing

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 50)
    logger.info("STEP 6: Exporting to CSV")
    logger.info("=" * 50)

    try:
        exporter = CSVExporter(
            data_dir=data_dir,
            output_csv=output_csv,
            resume=resume
        )

        processed, total = exporter.export(batch_size=batch_size)

        if processed > 0:
            # Print file size
            file_size = output_csv.stat().st_size / (1024 * 1024)
            logger.info(f"CSV file size: {file_size:.2f} MB")

        logger.info(f"Step 6 completed: Exported {processed}/{total} windows")
        return True

    except Exception as e:
        logger.error(f"Step 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Export fused embeddings and actions to CSV"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".data/pipeline_output",
        help="Path to dataset with processed windows"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=".data/vectordb_data.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing CSV file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for writing rows"
    )

    args = parser.parse_args()

    success = run_step6(
        data_dir=Path(args.data_dir),
        output_csv=Path(args.output_csv),
        resume=args.resume,
        batch_size=args.batch_size
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
