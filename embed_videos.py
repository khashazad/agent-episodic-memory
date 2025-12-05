import sys
from pathlib import Path

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

import mineclip.base
mineclip.base.VideoRewardBase = VideoRewardBase

from mineclip.mineclip import MineCLIP

def load_model(checkpoint_path: str, device: torch.device) -> MineCLIP:
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


def read_video_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.array(frames)


def resize_frames(frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    resized = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (target_size[1], target_size[0]))
        resized.append(resized_frame)
    return np.array(resized)


def create_chunks(frames: np.ndarray, chunk_size: int = 16) -> np.ndarray:

    n_frames = len(frames)
    n_chunks = n_frames // chunk_size
    if n_chunks == 0:
        padding = chunk_size - n_frames
        frames = np.concatenate([frames, np.tile(frames[-1:], (padding, 1, 1, 1))], axis=0)
        n_chunks = 1

    frames = frames[: n_chunks * chunk_size]
    chunks = frames.reshape(n_chunks, chunk_size, *frames.shape[1:])
    return chunks


@torch.no_grad()
def encode_video_chunks(
    model: MineCLIP,
    chunks: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    # tensor -> [N, 16, C, H, W]
    chunks_tensor = torch.from_numpy(chunks).permute(0, 1, 4, 2, 3).to(device)

    embeddings = []
    for i in range(0, len(chunks_tensor), batch_size):
        batch = chunks_tensor[i : i + batch_size]
        batch_embeddings = model.encode_video(batch)
        embeddings.append(batch_embeddings.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def process_video(
    video_path: Path,
    model: MineCLIP,
    device: torch.device,
    target_size: tuple[int, int] = (160, 256),
    chunk_size: int = 16,
    batch_size: int = 8,
) -> np.ndarray:
    frames = read_video_frames(str(video_path))

    if len(frames) == 0:
        return None

    frames = resize_frames(frames, target_size)

    chunks = create_chunks(frames, chunk_size)

    embeddings = encode_video_chunks(model, chunks, device, batch_size)

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".data/MineRLTreechop-v0",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=".ckpts/attn.pth",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="embedding.npy",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MineCLIP model...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully")

    data_dir = Path(args.data_dir)
    video_paths = list(data_dir.glob("*/recording.mp4"))
    print(f"{len(video_paths)} videos")

    for video_path in tqdm(video_paths, desc="Processing videos"):
        output_path = video_path.parent / args.output_name

        if output_path.exists():
            continue

        embeddings = process_video(
            video_path,
            model,
            device,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )

        if embeddings is not None:
            np.save(output_path, embeddings)

if __name__ == "__main__":
    main()
