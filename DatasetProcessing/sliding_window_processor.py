import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlidingWindowProcessor:
    """Processes MineRL episodes using overlapping sliding window chunking"""

    def __init__(self, data_dir: Path, output_dir: Path, window_size: int = 16, stride: int = 8):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.stride = stride
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized sliding window processor: window_size={window_size}, stride={stride}")
        logger.info(f"Overlap: {((window_size - stride) / window_size * 100):.1f}%")

    def get_episode_directories(self) -> List[Path]:
        episode_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(episode_dirs)} episodes in {self.data_dir}")
        return sorted(episode_dirs)

    def load_episode(self, episode_dir: Path) -> Optional[Tuple[np.ndarray, Dict]]:
        """Load video frames and action data from an episode directory."""
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
        """Generate overlapping sliding windows with given stride."""
        windows = []
        start = 0

        while start + self.window_size <= total_frames:
            end = start + self.window_size
            windows.append((start, end))
            start += self.stride

        logger.debug(f"Generated {len(windows)} windows for {total_frames} frames")
        return windows

    def format_actions(self, actions: Dict, start_idx: int, end_idx: int) -> Dict:
        """Convert action indices to human-readable descriptions for a window of actions."""
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
        """Process an episode using overlapping sliding windows."""
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

            window_dir = episode_output_dir / f"window_{window_idx:03d}"
            window_dir.mkdir(exist_ok=True)

            self._save_window_frames(frame_window, window_dir)
            self._save_window_actions(action_window, window_dir)

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

    def _save_window_frames(self, frames: np.ndarray, window_dir: Path):
        """Save window frames as individual images, video, and numpy array."""
        frames_dir = window_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(frames):
            frame_path = frames_dir / f"frame_{i:02d}.png"
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), frame_bgr)

        self._save_window_video(frames, window_dir / "window.mp4")
        np.save(window_dir / "frames.npy", frames)

    def _save_window_video(self, frames: np.ndarray, video_path: Path):
        """Save frame sequence as MP4 video."""
        height, width = frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    def _save_window_actions(self, action_window: Dict, window_dir: Path):
        """Save action data for the window."""
        with open(window_dir / "actions.txt", 'w') as f:
            for i, desc in enumerate(action_window['descriptions']):
                f.write(f"Step {i:2d}: {desc}\n")

        with open(window_dir / "actions.json", 'w') as f:
            json.dump(action_window, f, indent=2)

        np.save(window_dir / "actions.npy", action_window['raw_actions'])

    def process_all_episodes(self):
        """Process all episodes using sliding window approach."""
        episode_dirs = self.get_episode_directories()
        total_windows = 0

        for episode_dir in tqdm(episode_dirs, desc="Processing episodes with sliding windows"):
            try:
                windows_created = self.process_episode(episode_dir)
                total_windows += windows_created
            except Exception as e:
                logger.error(f"Error processing {episode_dir}: {e}")
                continue

        logger.info(f"Processing complete! Created {total_windows} total overlapping windows")

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


def main():
    parser = argparse.ArgumentParser(description="Process MineRL dataset using overlapping sliding window chunking")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".data/MineRLTreechop-v0",
        help="Path to MineRL dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".data/sliding_window_dataset",
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
        help="Number of frames to advance between windows (stride)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Limit number of episodes to process"
    )

    args = parser.parse_args()

    processor = SlidingWindowProcessor(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        window_size=args.window_size,
        stride=args.stride
    )

    if args.episodes:
        episode_dirs = processor.get_episode_directories()[:args.episodes]
        total_windows = 0
        for episode_dir in tqdm(episode_dirs, desc=f"Processing {args.episodes} episodes"):
            try:
                windows_created = processor.process_episode(episode_dir)
                total_windows += windows_created
            except Exception as e:
                logger.error(f"Error processing {episode_dir}: {e}")
                continue
        logger.info(f"Created {total_windows} windows from {args.episodes} episodes")
    else:
        processor.process_all_episodes()


if __name__ == "__main__":
    main()
