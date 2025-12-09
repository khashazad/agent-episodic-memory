#!/usr/bin/env python3
"""
This script processes full episode videos from MineRLTreechop-v0 and generates
natural language descriptions using Qwen2.5-VL. The same description is saved
to all chunk folders belonging to each episode.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prompt template for Minecraft video description
DESCRIPTION_PROMPT = """You are an expert at describing Minecraft gameplay videos.
Analyze the provided video frames and describe what's happening throughout this gameplay session.

Include in your description:
1. Player Actions: What is the player doing? (mining, building, exploring, chopping wood)
2. Environment: What location? What blocks/structures are visible?
3. Progress: What goal appears to be in progress?

Keep the description concise (2-3 sentences) and factual."""


def load_video_frames(video_path: Path, num_frames: int = 16) -> list[np.ndarray]:
    """Load and sample frames uniformly from a video file."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    # Use 90% of reported frames to avoid unreadable frames at end
    safe_end = int(total_frames * 0.9)
    frame_indices = np.linspace(0, safe_end, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()

    if len(frames) < num_frames // 2:
        raise ValueError(f"Could only read {len(frames)} frames from {video_path}")

    return frames


def frames_to_pil_images(frames: list[np.ndarray]) -> list[Image.Image]:
    return [Image.fromarray(frame) for frame in frames]


class DescriptionGenerator:
    """Generates video descriptions using a Vision-Language Model."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)


    def generate_description(self, frames: list[np.ndarray], max_new_tokens: int = 256) -> str:
        """Generate a description for the given video frames."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert frames to PIL images
        pil_images = frames_to_pil_images(frames)

        # Qwen2.5-VL expects images in the content as a list
        image_content = [{"type": "image", "image": img} for img in pil_images]

        messages = [
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": DESCRIPTION_PROMPT}],
            }
        ]

        # Process the input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=pil_images,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode the output
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        description = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return description.strip()


def get_episode_dirs(videos_dir: Path) -> list[Path]:
    """Get all episode directories from MineRLTreechop-v0."""
    episode_dirs = []

    for item in videos_dir.iterdir():
        if item.is_dir():
            # Check if it has a recording.mp4
            if (item / "recording.mp4").exists():
                episode_dirs.append(item)

    return sorted(episode_dirs)


def save_description_to_chunks(episode_name: str, description: str, chunks_dir: Path) -> int:
    """Save the same description to all chunks belonging to this episode.

    Returns the number of chunks updated.
    """
    episode_chunks_dir = chunks_dir / episode_name

    if not episode_chunks_dir.exists():
        logger.warning(f"No chunks directory found for episode: {episode_name}")
        return 0

    chunk_dirs = list(episode_chunks_dir.glob("chunk_*"))

    if not chunk_dirs:
        logger.warning(f"No chunk subdirectories found in: {episode_chunks_dir}")
        return 0

    count = 0
    for chunk_dir in chunk_dirs:
        if chunk_dir.is_dir():
            output_path = chunk_dir / "llm_derived_description.txt"
            with open(output_path, 'w') as f:
                f.write(description)
            count += 1

    return count


def check_episode_has_descriptions(episode_name: str, chunks_dir: Path) -> bool:
    """Check if an episode already has descriptions in its chunks."""
    episode_chunks_dir = chunks_dir / episode_name

    if not episode_chunks_dir.exists():
        return False

    # Check if at least one chunk has a description
    for chunk_dir in episode_chunks_dir.glob("chunk_*"):
        if chunk_dir.is_dir():
            desc_path = chunk_dir / "llm_derived_description.txt"
            if desc_path.exists():
                return True

    return False


def process_episodes(
    videos_dir: Path,
    chunks_dir: Path,
    generator: DescriptionGenerator,
    num_frames: int = 16,
    resume: bool = True,
    start_episode: int = 0,
    end_episode: Optional[int] = None,
):
    """Process episodes and generate descriptions."""
    episode_dirs = get_episode_dirs(videos_dir)

    if not episode_dirs:
        logger.error(f"No episodes found in {videos_dir}")
        return

    logger.info(f"Found {len(episode_dirs)} episodes")

    # Slice episodes based on start/end
    if end_episode is not None:
        episode_dirs = episode_dirs[start_episode:end_episode]
    else:
        episode_dirs = episode_dirs[start_episode:]

    logger.info(f"Processing episodes {start_episode} to {start_episode + len(episode_dirs)}")

    processed = 0
    skipped = 0
    failed = 0
    total_chunks = 0

    for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
        episode_name = episode_dir.name

        # Check if already processed (resume mode)
        if resume and check_episode_has_descriptions(episode_name, chunks_dir):
            skipped += 1
            continue

        try:
            # Load video frames
            video_path = episode_dir / "recording.mp4"
            frames = load_video_frames(video_path, num_frames)

            # Generate description
            description = generator.generate_description(frames)

            # Save to all chunks
            num_chunks = save_description_to_chunks(episode_name, description, chunks_dir)

            processed += 1
            total_chunks += num_chunks

            logger.debug(f"Processed {episode_name}: {num_chunks} chunks updated")

        except Exception as e:
            logger.error(f"Failed to process {episode_name}: {e}")
            failed += 1

    logger.info(f"Processing complete:")
    logger.info(f"  Processed: {processed} episodes")
    logger.info(f"  Skipped (already done): {skipped} episodes")
    logger.info(f"  Failed: {failed} episodes")
    logger.info(f"  Total chunks updated: {total_chunks}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text descriptions for Minecraft gameplay videos using a local VLM"
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=".data/MineRLTreechop-v0",
        help="Path to MineRLTreechop-v0 directory with episode videos"
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=".data/chunked_dataset_with_embeddings",
        help="Path to chunked dataset directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model ID for the VLM"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames to sample from each video"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip episodes that already have descriptions"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Process all episodes, even if they already have descriptions"
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Start from this episode index"
    )
    parser.add_argument(
        "--end-episode",
        type=int,
        default=None,
        help="End at this episode index (exclusive)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )

    args = parser.parse_args()

    # Handle resume flag
    resume = args.resume and not args.no_resume

    # Validate paths
    videos_dir = Path(args.videos_dir)
    chunks_dir = Path(args.chunks_dir)

    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        sys.exit(1)

    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        sys.exit(1)

    # Initialize generator
    generator = DescriptionGenerator(model_id=args.model, device=args.device)
    generator.load_model()

    # Process episodes
    process_episodes(
        videos_dir=videos_dir,
        chunks_dir=chunks_dir,
        generator=generator,
        num_frames=args.num_frames,
        resume=resume,
        start_episode=args.start_episode,
        end_episode=args.end_episode,
    )


if __name__ == "__main__":
    main()
