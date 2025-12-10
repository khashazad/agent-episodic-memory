#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import tarfile
import io

class VideoRewardBase(nn.Module):
    def __init__(self, *, image_encoder, temporal_encoder, reward_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.temporal_encoder = temporal_encoder
        self.reward_head = reward_head

import MineCLIP.mineclip.base
MineCLIP.mineclip.base.VideoRewardBase = VideoRewardBase

from MineCLIP.mineclip import MineCLIP
from MineCLIP.mineclip.tokenization import tokenize_batch

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

def load_text_description(chunk_dir: Path) -> str:
    """Load text description from chunk directory."""
    desc_path = chunk_dir / "llm_derived_description.txt"
    if desc_path.exists():
        with open(desc_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    raise FileNotFoundError(f"No llm_derived_description.txt found in {chunk_dir}")

@torch.no_grad()
def encode_text_batch(
    model: MineCLIP,
    texts: list[str],
    device: torch.device,
    max_length: int = 77,
) -> list[np.ndarray]:
    """Encode multiple text descriptions in batch."""
    if not texts:
        return []

    tokens = tokenize_batch(texts, max_length=max_length, language_model="clip")
    tokens = tokens.to(device)

    embeddings = model.encode_text(tokens)
    return embeddings.cpu().numpy()

def get_episodes_and_chunks(data_dir: Path) -> dict[Path, list[Path]]:
    """Get episodes and their chunk directories."""
    episodes = {}

    for episode_dir in data_dir.iterdir():
        if episode_dir.is_dir() and episode_dir.name.startswith("v3_"):
            chunk_dirs = []
            for chunk_dir in episode_dir.iterdir():
                if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk_"):
                    chunk_dirs.append(chunk_dir)
            if chunk_dirs:
                episodes[episode_dir] = sorted(chunk_dirs)

    return episodes

def process_episodes(episodes: dict[Path, list[Path]], model: MineCLIP, device: torch.device,
                                batch_size: int = 32, force_recompute: bool = False):
    """Process episodes by encoding unique descriptions once and copying to all chunks."""
    episodes_to_process = []

    # Find episodes that need processing
    for episode_dir, chunk_dirs in episodes.items():
        needs_processing = False
        for chunk_dir in chunk_dirs:
            text_embedding_path = chunk_dir / "text_embedding.npy"
            desc_path = chunk_dir / "llm_derived_description.txt"
            if desc_path.exists() and (force_recompute or not text_embedding_path.exists()):
                needs_processing = True
                break

        if needs_processing:
            episodes_to_process.append((episode_dir, chunk_dirs))

    if not episodes_to_process:
        print("All episodes already have text embeddings!")
        return

    print(f"Processing {len(episodes_to_process)} episodes")
    total_chunks = sum(len(chunks) for _, chunks in episodes_to_process)
    print(f"Total chunks to update: {total_chunks}")

    processed_episodes = 0
    processed_chunks = 0
    failed_episodes = 0

    # Process episodes in batches for efficient encoding
    unique_descriptions = []
    episode_desc_map = {}

    # First pass: collect unique descriptions
    for episode_dir, chunk_dirs in tqdm(episodes_to_process, desc="Collecting descriptions"):
        try:
            # Get description from first chunk (all chunks in episode have same description)
            first_chunk = chunk_dirs[0]
            desc_path = first_chunk / "llm_derived_description.txt"

            if desc_path.exists():
                with open(desc_path, 'r', encoding='utf-8') as f:
                    description = f.read().strip()

                if description not in unique_descriptions:
                    unique_descriptions.append(description)

                episode_desc_map[episode_dir] = description
            else:
                print(f"Warning: No description found for episode {episode_dir}")

        except Exception as e:
            print(f"Error processing episode {episode_dir}: {e}")
            failed_episodes += 1

    if not unique_descriptions:
        print("No valid descriptions found!")
        return

    print(f"Found {len(unique_descriptions)} unique descriptions to encode")

    # Second pass: encode unique descriptions in batches
    description_embeddings = {}

    for i in tqdm(range(0, len(unique_descriptions), batch_size), desc="Encoding descriptions"):
        batch_descriptions = unique_descriptions[i:i + batch_size]

        try:
            embeddings = encode_text_batch(model, batch_descriptions, device)

            for desc, embedding in zip(batch_descriptions, embeddings):
                description_embeddings[desc] = embedding.astype(np.float32)

        except Exception as e:
            print(f"Error encoding batch: {e}")
            failed_episodes += len(batch_descriptions)

    # Third pass: copy embeddings to all chunks
    for episode_dir, chunk_dirs in tqdm(episodes_to_process, desc="Copying embeddings to chunks"):
        if episode_dir not in episode_desc_map:
            continue

        description = episode_desc_map[episode_dir]
        if description not in description_embeddings:
            print(f"Warning: No embedding found for description in {episode_dir}")
            continue

        embedding = description_embeddings[description]

        try:
            for chunk_dir in chunk_dirs:
                text_embedding_path = chunk_dir / "text_embedding.npy"
                np.save(text_embedding_path, embedding)
                processed_chunks += 1

            processed_episodes += 1

        except Exception as e:
            print(f"Error saving embeddings for episode {episode_dir}: {e}")
            failed_episodes += 1

    print(f"Successfully processed {processed_episodes} episodes ({processed_chunks} chunks)")
    if failed_episodes > 0:
        print(f"Failed to process {failed_episodes} episodes")

def main():
    parser = argparse.ArgumentParser(description="Generate text embeddings for chunked dataset using MineCLIP")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".data/chunked_dataset_with_embeddings_and_llm_descriptions",
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
        default=64,
        help="Batch size for processing text descriptions"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute embeddings even if they already exist"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MineCLIP model...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully")

    print(f"\nProcessing dataset: {args.data_dir}")
    episodes = get_episodes_and_chunks(Path(args.data_dir))
    total_chunks = sum(len(chunks) for chunks in episodes.values())
    print(f"Found {len(episodes)} episodes with {total_chunks} total chunks")

    if episodes:
        process_episodes(
            episodes, model, device,
            batch_size=args.batch_size,
            force_recompute=args.force_recompute
        )

if __name__ == "__main__":
    main()
