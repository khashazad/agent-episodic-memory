#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

class VideoRewardBase(nn.Module):
    def __init__(self, *, image_encoder, temporal_encoder, reward_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.temporal_encoder = temporal_encoder
        self.reward_head = reward_head

import MineCLIP.mineclip.base
MineCLIP.mineclip.base.VideoRewardBase = VideoRewardBase

from MineCLIP.mineclip import MineCLIP

def load_model(checkpoint_path: str, device: torch.device) -> MineCLIP:
    """Load MineCLIP model from checkpoint."""
    model = MineCLIP(
        arch="vit_base_p16_fz.v2.t2",
        resolution=(160, 256),
        pool_type="attn.d2.nh8.glusw",
        image_feature_dim=512,
        mlp_adapter_spec="v0-2.t0",
        hidden_dim=512,
    ).to(device)

    model.load_ckpt(checkpoint_path, strict=False)
    model.eval()
    return model


def load_frames(chunk_dir: Path) -> np.ndarray:
    # Load from pre-saved frames.npy
    frames_path = chunk_dir / "frames.npy"
    if frames_path.exists():
        return np.load(frames_path)

    # Fallback to loading from MP4 or individual frames
    mp4_path = chunk_dir / "chunk.mp4"
    if mp4_path.exists():
        return load_frames_from_video(mp4_path)

    # Fallback to loading individual PNG frames
    frames_dir = chunk_dir / "frames"
    if frames_dir.exists():
        return load_frames_from_images(frames_dir)

    raise FileNotFoundError(f"No frame data found in {chunk_dir}")


def load_frames_from_video(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    # loop until end of video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.array(frames)


def load_frames_from_images(frames_dir: Path) -> np.ndarray:
    """Load frames from individual PNG files."""
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    frames = []

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    return np.array(frames)


def resize_frames(frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize frames to target resolution for MineCLIP."""
    resized = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (target_size[1], target_size[0]))
        resized.append(resized_frame)
    return np.array(resized)

@torch.no_grad()
def encode_chunk_batch(
    model: MineCLIP,
    chunk_frames_list: list[np.ndarray],
    device: torch.device,
    target_size: tuple[int, int] = (160, 256),
    batch_size: int = 8,
) -> list[np.ndarray]:
    """Encode multiple chunks in batches for efficiency."""
    embeddings = []

    for i in range(0, len(chunk_frames_list), batch_size):
        batch_chunks = chunk_frames_list[i:i + batch_size]

        # Resize all chunks in batch
        batch_resized = []
        for frames in batch_chunks:
            frames_resized = resize_frames(frames, target_size)
            batch_resized.append(frames_resized)

        # Stack into batch tensor: [B, 16, H, W, C] -> [B, 16, C, H, W]
        batch_tensor = torch.from_numpy(np.stack(batch_resized)).permute(0, 1, 4, 2, 3).to(device)

        # Encode batch
        batch_embeddings = model.encode_video(batch_tensor)
        embeddings.extend(batch_embeddings.cpu().numpy())

    return embeddings


def get_chunk_dirs(data_dir: Path) -> list[Path]:
    """Get all chunk directories from the dataset."""
    chunk_dirs = []

    for episode_dir in data_dir.iterdir():
        if episode_dir.is_dir():
            for chunk_dir in episode_dir.iterdir():
                if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk_"):
                    chunk_dirs.append(chunk_dir)

    return sorted(chunk_dirs)

def process_chunks_in_batches(chunk_dirs: list[Path], model: MineCLIP, device: torch.device,
                             batch_size: int = 8):
    """Process chunks in batches for better efficiency."""
    chunks_to_process = []
    for chunk_dir in chunk_dirs:
        embedding_path = chunk_dir / "embedding.npy"
        if not embedding_path.exists():
            chunks_to_process.append(chunk_dir)

    if not chunks_to_process:
        print("All chunks already have embeddings!")
        return

    print(f"Processing {len(chunks_to_process)} chunks in batches of {batch_size}")

    processed_count = 0
    for i in tqdm(range(0, len(chunks_to_process), batch_size), desc="Processing chunk batches"):
        batch_dirs = chunks_to_process[i:i + batch_size]

        # Load frames for batch
        batch_frames = []
        valid_dirs = []

        for chunk_dir in batch_dirs:
            try:
                frames = load_frames(chunk_dir)
                batch_frames.append(frames)
                valid_dirs.append(chunk_dir)
            except Exception as e:
                print(f"Error loading {chunk_dir}: {e}")

        if not batch_frames:
            continue

        # Encode batch
        try:
            embeddings = encode_chunk_batch(model, batch_frames, device, batch_size=len(batch_frames))

            # Save embeddings
            for chunk_dir, embedding in zip(valid_dirs, embeddings):
                embedding_path = chunk_dir / "embedding.npy"
                np.save(embedding_path, embedding)
                processed_count += 1

        except Exception as e:
            print(f"Error processing batch: {e}")

    print(f"Successfully processed {processed_count} chunks")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for chunked dataset using MineCLIP")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="chunked_dataset",
        help="Path to chunked dataset directory"
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
        help="Batch size for processing chunks"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute embeddings even if they already exist"
    )
    parser.add_argument(
        "--single-mode",
        action="store_true",
        help="Process chunks one by one instead of in batches"
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MineCLIP model...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully")

    # Get all chunk directories
    data_dir = Path(args.data_dir)
    chunk_dirs = get_chunk_dirs(data_dir)
    print(f"Found {len(chunk_dirs)} chunks to process")

    if not chunk_dirs:
        print(f"No chunks found in {data_dir}")
        return

    process_chunks_in_batches(
        chunk_dirs, model, device,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()
