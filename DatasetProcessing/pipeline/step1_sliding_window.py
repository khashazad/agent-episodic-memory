#!/usr/bin/env python3
"""
Step 1: Sliding Window Chunking

Creates overlapping sliding window chunks from MineRL episodes.
Each window contains 16 frames with a configurable stride (default: 8 frames).

Output per window:
- frames.npy: NumPy array of RGB frames [16, H, W, 3]
- actions.json: Action data for the window
- metadata.json: Window metadata (episode, index, frame range)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SlidingWindowProcessor:
    """Processes MineRL episodes using overlapping sliding window chunking."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        window_size: int = 16,
        stride: int = 8
    ):
        """Initialize the sliding window processor.

        Args:
            data_dir: Path to MineRL dataset directory
            output_dir: Output directory for processed windows
            window_size: Number of frames per window (default: 16)
            stride: Number of frames to advance between windows (default: 8)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.stride = stride
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized sliding window processor: window_size={window_size}, stride={stride}")
        logger.info(f"Overlap: {((window_size - stride) / window_size * 100):.1f}%")

    def get_episode_directories(self) -> List[Path]:
        """Get all episode directories from the dataset."""
        episode_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(episode_dirs)} episodes in {self.data_dir}")
        return sorted(episode_dirs)

    def load_episode(self, episode_dir: Path) -> Optional[Tuple[np.ndarray, Dict]]:
        """Load video frames and action data from an episode directory.

        Args:
            episode_dir: Path to episode directory

        Returns:
            Tuple of (frames array, actions dict) or None if loading fails
        """
        video_path = episode_dir / "recording.mp4"
        action_path = episode_dir / "rendered.npz"

        if not (video_path.exists() and action_path.exists()):
            logger.warning(f"Missing files in {episode_dir}")
            return None

        frames = self._load_video_frames(video_path)
        if frames is None:
            return None

        action_data = np.load(action_path)
        actions = {
            'forward': action_data['action$forward'],
            'left': action_data['action$left'],
            'right': action_data['action$right'],
            'back': action_data['action$back'],
            'jump': action_data['action$jump'],
            'sneak': action_data['action$sneak'],
            'sprint': action_data['action$sprint'],
            'attack': action_data['action$attack'],
            'camera': action_data['action$camera'],
            'reward': action_data['reward']
        }

        logger.debug(f"Episode {episode_dir.name}: {len(frames)} frames, {len(actions['forward'])} actions")
        return frames, actions

    def _load_video_frames(self, video_path: Path) -> Optional[np.ndarray]:
        """Load frames from video file."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            logger.warning(f"No frames loaded from {video_path}")
            return None

        return np.array(frames)

    def generate_windows(self, total_frames: int) -> List[Tuple[int, int]]:
        """Generate overlapping sliding windows with given stride.

        Args:
            total_frames: Total number of frames in episode

        Returns:
            List of (start_idx, end_idx) tuples for each window
        """
        windows = []
        start = 0

        while start + self.window_size <= total_frames:
            end = start + self.window_size
            windows.append((start, end))
            start += self.stride

        logger.debug(f"Generated {len(windows)} windows for {total_frames} frames")
        return windows

    def format_actions(self, actions: Dict, start_idx: int, end_idx: int) -> Dict:
        """Format action data for a window.

        Args:
            actions: Full episode actions dict
            start_idx: Start frame index
            end_idx: End frame index

        Returns:
            Formatted actions dict with raw actions, descriptions, and summary
        """
        chunk_actions = {}

        for key in actions:
            if key == 'camera':
                chunk_actions[key] = actions[key][start_idx:end_idx].tolist()
            else:
                chunk_actions[key] = actions[key][start_idx:end_idx].tolist()

        descriptions = []
        for i in range(len(chunk_actions['forward'])):
            step_actions = []

            if chunk_actions['forward'][i]:
                step_actions.append("forward")
            if chunk_actions['back'][i]:
                step_actions.append("back")
            if chunk_actions['left'][i]:
                step_actions.append("left")
            if chunk_actions['right'][i]:
                step_actions.append("right")

            if chunk_actions['jump'][i]:
                step_actions.append("jump")
            if chunk_actions['sneak'][i]:
                step_actions.append("sneak")
            if chunk_actions['sprint'][i]:
                step_actions.append("sprint")
            if chunk_actions['attack'][i]:
                step_actions.append("attack")

            camera_x, camera_y = chunk_actions['camera'][i]
            if abs(camera_x) > 0.1 or abs(camera_y) > 0.1:
                step_actions.append(f"camera({camera_x:.1f}, {camera_y:.1f})")

            if chunk_actions['reward'][i] > 0:
                step_actions.append(f"reward={chunk_actions['reward'][i]:.1f}")

            if step_actions:
                descriptions.append(" + ".join(step_actions))
            else:
                descriptions.append("idle")

        return {
            'raw_actions': chunk_actions,
            'descriptions': descriptions,
            'summary': {
                'total_steps': len(descriptions),
                'movement_steps': sum(1 for d in descriptions if any(m in d for m in ['forward', 'back', 'left', 'right'])),
                'attack_steps': sum(1 for d in descriptions if 'attack' in d),
                'camera_steps': sum(1 for d in descriptions if 'camera' in d),
                'total_reward': sum(chunk_actions['reward'])
            }
        }

    def process_episode(self, episode_dir: Path) -> int:
        """Process an episode using overlapping sliding windows.

        Args:
            episode_dir: Path to episode directory

        Returns:
            Number of windows created
        """
        episode_data = self.load_episode(episode_dir)
        if episode_data is None:
            return 0

        frames, actions = episode_data
        total_frames = len(frames)

        if total_frames < self.window_size:
            logger.warning(f"Episode {episode_dir.name} too short for windows ({total_frames} < {self.window_size})")
            return 0

        windows = self.generate_windows(total_frames)
        episode_output_dir = self.output_dir / episode_dir.name
        episode_output_dir.mkdir(exist_ok=True)

        windows_created = 0
        for window_idx, (start_frame, end_frame) in enumerate(windows):
            frame_window = frames[start_frame:end_frame]
            action_window = self.format_actions(actions, start_frame, end_frame)

            window_dir = episode_output_dir / f"window_{window_idx:04d}"
            window_dir.mkdir(exist_ok=True)

            # Save frames as numpy array
            np.save(window_dir / "frames.npy", frame_window)

            # Save actions
            with open(window_dir / "actions.json", 'w') as f:
                json.dump(action_window, f, indent=2)

            # Save metadata
            metadata = {
                'episode': episode_dir.name,
                'window_index': window_idx,
                'frame_range': [start_frame, end_frame],
                'window_size': self.window_size,
                'stride': self.stride,
                'overlap_frames': self.window_size - self.stride,
                'action_summary': action_window['summary']
            }

            with open(window_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            windows_created += 1

        logger.info(f"Created {windows_created} overlapping windows for episode {episode_dir.name}")
        return windows_created

    def process_all_episodes(self, max_episodes: Optional[int] = None) -> int:
        """Process all episodes using sliding window approach.

        Args:
            max_episodes: Optional limit on number of episodes to process

        Returns:
            Total number of windows created
        """
        episode_dirs = self.get_episode_directories()

        if max_episodes:
            episode_dirs = episode_dirs[:max_episodes]
            logger.info(f"Processing {max_episodes} episodes (limited)")

        total_windows = 0

        for episode_dir in tqdm(episode_dirs, desc="Processing episodes with sliding windows"):
            try:
                windows_created = self.process_episode(episode_dir)
                total_windows += windows_created
            except Exception as e:
                logger.error(f"Error processing {episode_dir}: {e}")
                continue

        logger.info(f"Processing complete! Created {total_windows} total overlapping windows")

        # Save summary
        summary = {
            'total_episodes': len(episode_dirs),
            'total_windows': total_windows,
            'window_size': self.window_size,
            'stride': self.stride,
            'overlap_percentage': ((self.window_size - self.stride) / self.window_size * 100),
            'output_directory': str(self.output_dir),
            'method': 'overlapping_sliding_window'
        }

        with open(self.output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        return total_windows


def run_step1(
    input_dir: Path,
    output_dir: Path,
    window_size: int = 16,
    stride: int = 8,
    max_episodes: Optional[int] = None
) -> bool:
    """Run Step 1: Sliding window chunk creation.

    Args:
        input_dir: Path to MineRL dataset directory
        output_dir: Output directory for processed windows
        window_size: Number of frames per window
        stride: Number of frames between window starts
        max_episodes: Optional limit on episodes to process

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 50)
    logger.info("STEP 1: Creating sliding window chunks")
    logger.info("=" * 50)

    try:
        processor = SlidingWindowProcessor(
            data_dir=input_dir,
            output_dir=output_dir,
            window_size=window_size,
            stride=stride
        )

        total_windows = processor.process_all_episodes(max_episodes=max_episodes)

        logger.info(f"Step 1 completed: Created {total_windows} windows")
        return True

    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Create sliding window chunks from MineRL episodes"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".data/MineRLTreechop-v0",
        help="Path to MineRL dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".data/pipeline_output",
        help="Output directory for processed windows"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Number of frames per window"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Number of frames to advance between windows"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episodes to process"
    )

    args = parser.parse_args()

    success = run_step1(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        window_size=args.window_size,
        stride=args.stride,
        max_episodes=args.max_episodes
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
