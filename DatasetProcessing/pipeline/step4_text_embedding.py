#!/usr/bin/env python3
"""
Step 4: Text Embedding

Encodes scene descriptions using MineCLIP's text encoder.
Produces a 512-dimensional embedding vector for each description.

Input: description.txt (natural language description)
Output: text_embedding.npy (512-dim vector)
"""

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from . import get_device, get_window_dirs, load_mineclip_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_text_batch(
    model,
    texts: List[str],
    device: torch.device
) -> List[np.ndarray]:
    """Encode multiple text descriptions in a batch.

    Args:
        model: MineCLIP model
        texts: List of text descriptions
        device: Torch device

    Returns:
        List of embedding arrays, each [512]
    """
    # MineCLIP's encode_text handles tokenization internally
    # It accepts strings or list of strings
    embeddings = model.clip_model.encode_text(texts)

    return [emb.cpu().numpy() for emb in embeddings]


def process_windows_text(
    window_dirs: List[Path],
    model,
    device: torch.device,
    batch_size: int = 32,
    force_recompute: bool = False
) -> int:
    """Process windows to generate text embeddings.

    Args:
        window_dirs: List of window directories to process
        model: MineCLIP model
        device: Torch device
        batch_size: Number of descriptions per batch
        force_recompute: If True, recompute existing embeddings

    Returns:
        Number of windows successfully processed
    """
    # Collect windows that need processing and have descriptions
    windows_to_process = []
    descriptions_to_encode = []

    for window_dir in window_dirs:
        embedding_path = window_dir / "text_embedding.npy"
        description_path = window_dir / "description.txt"

        # Skip if embedding exists and not force recompute
        if not force_recompute and embedding_path.exists():
            continue

        # Skip if no description
        if not description_path.exists():
            continue

        # Read description
        try:
            with open(description_path, 'r', encoding='utf-8') as f:
                description = f.read().strip()

            if description:
                windows_to_process.append(window_dir)
                descriptions_to_encode.append(description)
        except Exception as e:
            logger.warning(f"Error reading description from {window_dir}: {e}")

    if not windows_to_process:
        logger.info("No windows need text embedding processing!")
        return 0

    logger.info(f"Processing {len(windows_to_process)} descriptions in batches of {batch_size}")

    processed_count = 0

    for i in tqdm(range(0, len(windows_to_process), batch_size), desc="Encoding text batches"):
        batch_dirs = windows_to_process[i:i + batch_size]
        batch_texts = descriptions_to_encode[i:i + batch_size]

        try:
            # Encode batch
            embeddings = encode_text_batch(model, batch_texts, device)

            # Save embeddings
            for window_dir, embedding in zip(batch_dirs, embeddings):
                embedding_path = window_dir / "text_embedding.npy"
                np.save(embedding_path, embedding)
                processed_count += 1

        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            # Try one by one for this batch
            for window_dir, text in zip(batch_dirs, batch_texts):
                try:
                    embedding = encode_text_batch(model, [text], device)[0]
                    embedding_path = window_dir / "text_embedding.npy"
                    np.save(embedding_path, embedding)
                    processed_count += 1
                except Exception as inner_e:
                    logger.error(f"Error encoding text for {window_dir}: {inner_e}")

    return processed_count


def run_step4(
    data_dir: Path,
    checkpoint_path: Path,
    batch_size: int = 32,
    force_recompute: bool = False
) -> bool:
    """Run Step 4: Text embedding generation.

    Args:
        data_dir: Path to dataset with window directories
        checkpoint_path: Path to MineCLIP checkpoint
        batch_size: Batch size for processing
        force_recompute: If True, recompute existing embeddings

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 50)
    logger.info("STEP 4: Generating text embeddings")
    logger.info("=" * 50)

    try:
        device = get_device()
        logger.info(f"Using device: {device}")

        logger.info("Loading MineCLIP model...")
        model = load_mineclip_model(str(checkpoint_path), device)
        logger.info("Model loaded successfully")

        window_dirs = get_window_dirs(data_dir)
        logger.info(f"Found {len(window_dirs)} windows")

        if not window_dirs:
            logger.warning(f"No windows found in {data_dir}")
            return True

        processed = process_windows_text(
            window_dirs, model, device,
            batch_size=batch_size,
            force_recompute=force_recompute
        )

        logger.info(f"Step 4 completed: Processed {processed} text embeddings")
        return True

    except Exception as e:
        logger.error(f"Step 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Generate text embeddings using MineCLIP"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".data/pipeline_output",
        help="Path to dataset with window directories"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=".ckpts/attn.pth",
        help="Path to MineCLIP checkpoint"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing descriptions"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute embeddings even if they already exist"
    )

    args = parser.parse_args()

    success = run_step4(
        data_dir=Path(args.data_dir),
        checkpoint_path=Path(args.checkpoint),
        batch_size=args.batch_size,
        force_recompute=args.force_recompute
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
