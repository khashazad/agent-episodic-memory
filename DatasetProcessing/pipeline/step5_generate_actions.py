#!/usr/bin/env python3
"""
Step 5: Generate Actions

Uses Qwen2.5-VL Vision-Language Model to predict the next best action
for the agent to take based on the video window content.

Input: frames.npy (16 RGB frames)
Output: next_action.json (JSON action dict)

Action format:
{
    "forward": 0 or 1,
    "back": 0 or 1,
    "left": 0 or 1,
    "right": 0 or 1,
    "jump": 0 or 1,
    "attack": 0 or 1,
    "camera": [pitch_delta, yaw_delta]
}
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from . import get_window_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Default action template
ACTION_TEMPLATE = {
    "forward": 0,
    "back": 0,
    "left": 0,
    "right": 0,
    "jump": 0,
    "attack": 0,
    "camera": [0.0, 0.0]
}


class ActionGenerator:
    """Generates next action predictions using Qwen2.5-VL."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto"
    ):
        """Initialize the action generator.

        Args:
            model_id: HuggingFace model ID for Qwen VLM
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the Qwen VLM model and processor."""
        logger.info(f"Loading model: {self.model_id}")

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "Please install transformers with: pip install transformers>=4.37.0"
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Determine device and dtype
        if self.device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
            else:
                device_map = "cpu"
                torch_dtype = torch.float32
        else:
            device_map = self.device
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        logger.info(f"Model loaded on device: {device_map}")

    def sample_frames(self, frames: np.ndarray, num_frames: int = 4) -> List[Image.Image]:
        """Sample frames evenly from the window for VLM input.

        Args:
            frames: NumPy array of frames [N, H, W, 3]
            num_frames: Number of frames to sample

        Returns:
            List of PIL Images
        """
        total_frames = len(frames)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        sampled = []
        for idx in indices:
            img = Image.fromarray(frames[idx].astype(np.uint8))
            sampled.append(img)

        return sampled

    def parse_action_json(self, response: str) -> Optional[Dict]:
        """Parse JSON action dict from model response.

        Args:
            response: Raw model response string

        Returns:
            Parsed action dict or None if parsing fails
        """
        # Try to find JSON in the response
        # Pattern 1: Direct JSON object
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, response)

        for match in matches:
            try:
                action = json.loads(match)
                # Validate it has expected keys
                valid_keys = {"forward", "back", "left", "right", "jump", "attack", "camera"}
                if any(k in valid_keys for k in action.keys()):
                    # Fill in missing keys with defaults
                    result = ACTION_TEMPLATE.copy()
                    for key, value in action.items():
                        if key in result:
                            result[key] = value
                    return result
            except json.JSONDecodeError:
                continue

        return None

    def generate_action(self, frames: np.ndarray) -> Dict:
        """Generate next action prediction for a window of frames.

        Args:
            frames: NumPy array of frames [16, H, W, 3]

        Returns:
            Action dict with movement and camera controls
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Sample 8 frames from the window
        images = self.sample_frames(frames, num_frames=8)

        # Build the prompt with chain-of-thought reasoning (no biased example)
        prompt = """You are an AI agent playing Minecraft. Your goal is to collect wood from trees.

Look at these 8 sequential frames from your current view carefully and analyze them.

First, answer these questions:
1. What do you see in the frames? Describe the environment (grass, trees, sky, blocks, etc.)
2. Is there a tree visible? If yes, where is it positioned (center, left, right, far, close)?
3. Are you currently facing a tree trunk or looking elsewhere?
4. What should be your next action to efficiently collect wood?

Available actions:
- forward: 0 or 1 (move forward)
- back: 0 or 1 (move backward)
- left: 0 or 1 (strafe left)
- right: 0 or 1 (strafe right)
- jump: 0 or 1 (jump)
- attack: 0 or 1 (break blocks/attack - use to chop wood)
- camera: [pitch, yaw] where pitch is vertical (-10 to 10, negative=look up) and yaw is horizontal (-10 to 10, negative=look left)

Decision guidelines:
- If a tree trunk is directly in front and close: attack=1, forward=0
- If a tree is visible but far: forward=1, attack=0
- If a tree is to the left: camera=[0, -5] to turn left
- If a tree is to the right: camera=[0, 5] to turn right
- If no tree visible: camera=[0, 10] to look around
- If looking at ground: camera=[-5, 0] to look up
- If looking at sky: camera=[5, 0] to look down

After your analysis, output your chosen action as a JSON object on a new line with exactly these keys:
{"forward": <0 or 1>, "back": <0 or 1>, "left": <0 or 1>, "right": <0 or 1>, "jump": <0 or 1>, "attack": <0 or 1>, "camera": [<pitch>, <yaw>]}

Your analysis and action:"""

        # Create message with images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "image", "image": images[2]},
                    {"type": "image", "image": images[3]},
                    {"type": "image", "image": images[4]},
                    {"type": "image", "image": images[5]},
                    {"type": "image", "image": images[6]},
                    {"type": "image", "image": images[7]},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = inputs.to(self.model.device)

        # Generate with sampling for diverse outputs
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,  # More tokens for chain-of-thought reasoning
                do_sample=True,      # Enable sampling for diversity
                temperature=0.7,     # Add randomness
                top_p=0.9,           # Nucleus sampling
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Log raw output for debugging
        logger.info(f"Raw model output: {output[:500]}")

        # Parse the action
        action = self.parse_action_json(output)

        logger.info(f"Parsed action: {action}")

        if action is None:
            logger.warning(f"Could not parse action from response: {output[:200]}")
            # Return default action (move forward and look around)
            return {"forward": 1, "back": 0, "left": 0, "right": 0,
                    "jump": 0, "attack": 0, "camera": [0.0, 5.0]}

        return action

def process_windows(
    data_dir: Path,
    generator: ActionGenerator,
    resume: bool = True,
    start_window: int = 0,
    end_window: Optional[int] = None
) -> int:
    """Process windows to generate action predictions.

    Args:
        data_dir: Path to dataset with window directories
        generator: ActionGenerator instance
        resume: If True, skip windows that already have actions
        start_window: Start from this window index
        end_window: End at this window index (exclusive)

    Returns:
        Number of windows processed
    """
    window_dirs = get_window_dirs(data_dir)

    if not window_dirs:
        logger.warning(f"No windows found in {data_dir}")
        return 0

    # Apply range limits
    if end_window:
        window_dirs = window_dirs[start_window:end_window]
    else:
        window_dirs = window_dirs[start_window:]

    logger.info(f"Processing {len(window_dirs)} windows for action predictions")

    processed_count = 0

    for window_dir in tqdm(window_dirs, desc="Generating actions"):
        action_path = window_dir / "next_action.json"

        # Skip if already exists and resume is True
        if resume and action_path.exists():
            continue

        try:
            # Load frames
            frames_path = window_dir / "frames.npy"
            if not frames_path.exists():
                logger.warning(f"No frames.npy in {window_dir}")
                continue

            frames = np.load(frames_path)

            # Generate action from frames only
            action = generator.generate_action(frames)

            # Save action
            with open(action_path, 'w', encoding='utf-8') as f:
                json.dump(action, f, indent=2)

            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing {window_dir}: {e}")
            continue

    return processed_count


def run_step5(
    data_dir: Path,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "auto",
    resume: bool = True,
    start_window: int = 0,
    end_window: Optional[int] = None
) -> bool:
    """Run Step 5: Generate action predictions using Qwen VLM.

    Args:
        data_dir: Path to dataset with window directories
        model_id: HuggingFace model ID for Qwen VLM
        device: Device to use ('auto', 'cuda', 'cpu')
        resume: If True, skip windows that already have actions
        start_window: Start from this window index
        end_window: End at this window index

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 50)
    logger.info("STEP 5: Generating action predictions")
    logger.info("=" * 50)

    try:
        generator = ActionGenerator(model_id=model_id, device=device)
        generator.load_model()

        processed = process_windows(
            data_dir=data_dir,
            generator=generator,
            resume=resume,
            start_window=start_window,
            end_window=end_window
        )

        logger.info(f"Step 5 completed: Generated {processed} action predictions")
        return True

    except Exception as e:
        logger.error(f"Step 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Generate action predictions using Qwen VLM"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".data/pipeline_output",
        help="Path to dataset with window directories"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID for Qwen VLM"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip windows that already have actions"
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
        help="End at this window index"
    )

    args = parser.parse_args()

    success = run_step5(
        data_dir=Path(args.data_dir),
        model_id=args.model_id,
        device=args.device,
        resume=not args.no_resume,
        start_window=args.start_window,
        end_window=args.end_window
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
