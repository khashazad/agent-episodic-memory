#!/usr/bin/env python3
"""
Step 2: Video Embedding

Encodes 16-frame windows using MineCLIP video encoder.
Produces a 512-dimensional embedding vector for each window.

Input: frames.npy (16 RGB frames)
Output: video_embedding.npy (512-dim vector)
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from . import get_device, get_window_dirs, load_mineclip_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_frames(window_dir: Path) -> np.ndarray:
    """Load frames from a window directory.

    Args:
        window_dir: Path to window directory containing frames.npy

    Returns:
        NumPy array of RGB frames [16, H, W, 3]
    """
    frames_path = window_dir / "frames.npy"
    if frames_path.exists():
        return np.load(frames_path)

    raise FileNotFoundError(f"No frames.npy found in {window_dir}")


def resize_frames(frames: np.ndarray, target_size: tuple = (160, 256)) -> np.ndarray:
    """Resize frames to target resolution for MineCLIP.

    Args:
        frames: Input frames [N, H, W, 3]
        target_size: Target (height, width) for MineCLIP

    Returns:
        Resized frames array
    """
    resized = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (target_size[1], target_size[0]))
        resized.append(resized_frame)
    return np.array(resized)


@torch.no_grad()
def encode_window_batch(
    model,
    window_frames_list: List[np.ndarray],
    device: torch.device,
    target_size: tuple = (160, 256),
) -> List[np.ndarray]:
    """Encode multiple windows in a single batch.

    Args:
        model: MineCLIP model
        window_frames_list: List of frame arrays, each [16, H, W, 3]
        device: Torch device
        target_size: Target resolution for MineCLIP

    Returns:
        List of embedding arrays, each [512]
    """
    # Resize all windows
    batch_resized = []
    for frames in window_frames_list:
        frames_resized = resize_frames(frames, target_size)
        batch_resized.append(frames_resized)

    # Stack into batch tensor: [B, 16, H, W, C] -> [B, 16, C, H, W]
    batch_tensor = torch.from_numpy(np.stack(batch_resized)).permute(0, 1, 4, 2, 3).to(device)

    # Encode batch
    batch_embeddings = model.encode_video(batch_tensor)

    return [emb.cpu().numpy() for emb in batch_embeddings]


def process_windows_batch(
    window_dirs: List[Path],
    model,
    device: torch.device,
    batch_size: int = 8,
    force_recompute: bool = False
) -> int:
    """Process windows in batches for efficiency.

    Args:
        window_dirs: List of window directories to process
        model: MineCLIP model
        device: Torch device
        batch_size: Number of windows per batch
        force_recompute: If True, recompute existing embeddings

    Returns:
        Number of windows successfully processed
    """
    # Filter windows that need processing
    windows_to_process = []
    for window_dir in window_dirs:
        embedding_path = window_dir / "video_embedding.npy"
        if force_recompute or not embedding_path.exists():
            windows_to_process.append(window_dir)

    if not windows_to_process:
        logger.info("All windows already have video embeddings!")
        return 0

    logger.info(f"Processing {len(windows_to_process)} windows in batches of {batch_size}")

    processed_count = 0

    for i in tqdm(range(0, len(windows_to_process), batch_size), desc="Encoding video batches"):
        batch_dirs = windows_to_process[i:i + batch_size]

        # Load frames for batch
        batch_frames = []
        valid_dirs = []

        for window_dir in batch_dirs:
            try:
                frames = load_frames(window_dir)
                batch_frames.append(frames)
                valid_dirs.append(window_dir)
            except Exception as e:
                logger.warning(f"Error loading {window_dir}: {e}")

        if not batch_frames:
            continue

        # Encode batch
        try:
            embeddings = encode_window_batch(model, batch_frames, device)

            # Save embeddings
            for window_dir, embedding in zip(valid_dirs, embeddings):
                embedding_path = window_dir / "video_embedding.npy"
                np.save(embedding_path, embedding)
                processed_count += 1

        except Exception as e:
            logger.error(f"Error encoding batch: {e}")

    return processed_count


def run_step2(
    data_dir: Path,
    checkpoint_path: Path,
    batch_size: int = 16,
    force_recompute: bool = False
) -> bool:
    """Run Step 2: Video embedding generation.

    Args:
        data_dir: Path to dataset with window directories
        checkpoint_path: Path to MineCLIP checkpoint
        batch_size: Batch size for processing
        force_recompute: If True, recompute existing embeddings

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 50)
    logger.info("STEP 2: Generating video embeddings")
    logger.info("=" * 50)

    try:
        device = get_device()
        logger.info(f"Using device: {device}")

        logger.info("Loading MineCLIP model...")
        model = load_mineclip_model(str(checkpoint_path), device)
        logger.info("Model loaded successfully")

        window_dirs = get_window_dirs(data_dir)
        logger.info(f"Found {len(window_dirs)} windows to process")

        if not window_dirs:
            logger.warning(f"No windows found in {data_dir}")
            return True

        processed = process_windows_batch(
            window_dirs, model, device,
            batch_size=batch_size,
            force_recompute=force_recompute
        )

        logger.info(f"Step 2 completed: Processed {processed} windows")
        return True

    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Generate video embeddings using MineCLIP"
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
        default=16,
        help="Batch size for processing windows"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute embeddings even if they already exist"
    )

    args = parser.parse_args()

    success = run_step2(
        data_dir=Path(args.data_dir),
        checkpoint_path=Path(args.checkpoint),
        batch_size=args.batch_size,
        force_recompute=args.force_recompute
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
