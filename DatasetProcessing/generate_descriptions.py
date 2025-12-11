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
DESCRIPTION_PROMPT = """
### SYSTEM PROMPT

You are an expert Minecraft AI researcher. Your task is to watch a short clip of gameplay (approx 0.8s) and **reverse-engineer the specific player controls** that caused the visual movement.

You must ignore the aesthetics of the biome and focus entirely on **Agent Dynamics**.

**1. Visual Cues & Translation Rules:**
Scan the video for these specific indicators to determine the action:

* **The "Hand Sway" (Right side of screen):**
    * *Rhythmic swinging/punching animation:* -> **Action: ATTACK/CHOP**
    * *Static hand:* -> **Action: NO ATTACK**
    * *Item moving up/down:* -> **Action: INTERACT/PLACE**
* **Camera Motion (Optical Flow):**
    * *World moving toward camera:* -> **Action: MOVE FORWARD**
    * *Rhythmic vertical bobbing:* -> **Action: WALKING/SPRINTING** (If bobbing is fast, it is SPRINTING).
    * *Sudden vertical rise:* -> **Action: JUMP**
* **Targeting (Crosshair):**
    * *Cracks appearing on a block:* -> **Action: ACTIVELY CHOPPING**
    * *Highlight/Outline on a block:* -> **Action: TARGETING**

**2. Analysis Process (Chain of Thought):**
Before answering, briefly analyze:
1.  **Movement:** Is the agent stationary, walking, or sprinting?
2.  **Interaction:** Is the arm swinging? Are particles visible?
3.  **Intent:** Is the agent trying to reach a tree, or are they currently harvesting it?

**3. Output Format:**
Provide a single, concise **Imperative Instruction** that describes the intent.
* *Format:* `[Verb] [Object/Direction]`
* *Constraint:* Max 10 words.
* *Forbidden:* Do not describe the scenery ("I see a forest"). Only describe the *action*.

**Few-Shot Examples:**
* *Video shows:* Camera moving fast towards a tree, hand is still.
    * **Output:** "Sprint forward towards the tree."
* *Video shows:* Camera is stationary in front of wood, hand is swinging, cracks appear on wood.
    * **Output:** "Hold attack to chop the log."
* *Video shows:* Camera looks up 45 degrees, hand swings at leaves.
    * **Output:** "Look up and clear the leaves."
* *Video shows:* Camera spins right, world blurs.
    * **Output:** "Turn right quickly to scan surroundings."
"""


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


def get_window_dirs(data_dir: Path) -> list[Path]:
    """Get all window directories from the sliding window dataset."""
    window_dirs = []

    for episode_dir in data_dir.iterdir():
        if episode_dir.is_dir():
            for window_dir in episode_dir.iterdir():
                if window_dir.is_dir() and (window_dir.name.startswith("window_") or window_dir.name.startswith("chunk_")):
                    window_dirs.append(window_dir)

    return sorted(window_dirs)


def load_window_frames(window_dir: Path) -> list[np.ndarray]:
    """Load frames from a sliding window directory."""
    # Try loading from pre-saved frames.npy first
    frames_path = window_dir / "frames.npy"
    if frames_path.exists():
        frames_array = np.load(frames_path)
        return [frame for frame in frames_array]

    # Fallback to loading from video file
    # Try window.mp4 first (sliding window format)
    video_path = window_dir / "window.mp4"
    if video_path.exists():
        return load_video_frames_from_file(video_path)

    # Fallback to chunk.mp4 (backwards compatibility)
    video_path = window_dir / "chunk.mp4"
    if video_path.exists():
        return load_video_frames_from_file(video_path)

    # Fallback to individual frame files
    frames_dir = window_dir / "frames"
    if frames_dir.exists():
        return load_frames_from_images(frames_dir)

    raise FileNotFoundError(f"No frame data found in {window_dir}")


def load_video_frames_from_file(video_path: Path) -> list[np.ndarray]:
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    return frames


def load_frames_from_images(frames_dir: Path) -> list[np.ndarray]:
    """Load frames from individual image files."""
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    frames = []

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    if len(frames) == 0:
        raise ValueError(f"No frames found in {frames_dir}")

    return frames


def save_description_to_window(window_dir: Path, description: str) -> bool:
    """Save description to a single window directory.

    Returns True if successful, False otherwise.
    """
    try:
        output_path = window_dir / "llm_derived_description.txt"
        with open(output_path, 'w') as f:
            f.write(description)
        return True
    except Exception as e:
        logger.error(f"Failed to save description to {window_dir}: {e}")
        return False


def check_window_has_description(window_dir: Path) -> bool:
    """Check if a window already has a description."""
    desc_path = window_dir / "llm_derived_description.txt"
    return desc_path.exists()


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


def process_windows(
    data_dir: Path,
    generator: DescriptionGenerator,
    resume: bool = True,
    start_window: int = 0,
    end_window: Optional[int] = None,
):
    """Process individual windows and generate descriptions."""
    window_dirs = get_window_dirs(data_dir)

    if not window_dirs:
        logger.error(f"No windows found in {data_dir}")
        return

    logger.info(f"Found {len(window_dirs)} windows")

    # Slice windows based on start/end
    if end_window is not None:
        window_dirs = window_dirs[start_window:end_window]
    else:
        window_dirs = window_dirs[start_window:]

    logger.info(f"Processing windows {start_window} to {start_window + len(window_dirs)}")

    processed = 0
    skipped = 0
    failed = 0

    for window_dir in tqdm(window_dirs, desc="Processing windows"):
        # Check if already processed (resume mode)
        if resume and check_window_has_description(window_dir):
            skipped += 1
            continue

        try:
            # Load window frames
            frames = load_window_frames(window_dir)

            # Generate description
            description = generator.generate_description(frames)

            # Save description
            if save_description_to_window(window_dir, description):
                processed += 1
                logger.debug(f"Processed window: {window_dir.name}")
            else:
                failed += 1

        except Exception as e:
            logger.error(f"Failed to process {window_dir}: {e}")
            failed += 1

    logger.info(f"Processing complete:")
    logger.info(f"  Processed: {processed} windows")
    logger.info(f"  Skipped (already done): {skipped} windows")
    logger.info(f"  Failed: {failed} windows")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text descriptions for sliding window chunks using a local VLM"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="sliding_window_dataset",
        help="Path to sliding window dataset directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model ID for the VLM"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip windows that already have descriptions"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Process all windows, even if they already have descriptions"
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=0,
        help="Start from this window index"
    )
    parser.add_argument(
        "--end-window",
        type=int,
        default=None,
        help="End at this window index (exclusive)"
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
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Initialize generator
    generator = DescriptionGenerator(model_id=args.model, device=args.device)
    generator.load_model()

    # Process windows
    process_windows(
        data_dir=data_dir,
        generator=generator,
        resume=resume,
        start_window=args.start_window,
        end_window=args.end_window,
    )


if __name__ == "__main__":
    main()
