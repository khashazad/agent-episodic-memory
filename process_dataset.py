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


"""Processes MineRL episodes into 16-frame chunks"""
class MineRLProcessor:

    def __init__(self, data_dir: Path, output_dir: Path, chunk_size: int = 16):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_episode_directories(self) -> List[Path]:

        episode_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        logger.info(f"Found {len(episode_dirs)} episodes in {self.data_dir}")

        return sorted(episode_dirs)

    """Load video frames and action data from an episode directory."""
    def load_episode(self, episode_dir: Path) -> Optional[Tuple[np.ndarray, Dict]]:
        video_path = episode_dir / "recording.mp4"
        action_path = episode_dir / "rendered.npz"

        if not (video_path.exists() and action_path.exists()):
            logger.warning(f"Missing files in {episode_dir}")
            return None

        # Load video frames
        frames = self._load_video_frames(video_path)
        if frames is None:
            return None

        # Load action data
        action_data = np.load(action_path)

        # Convert to dictionary for easier handling
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

        # loop until end of video
        while True:
            ret, frame = cap.read()

            # end of video
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            logger.warning(f"No frames loaded from {video_path}")
            return None

        return np.array(frames)

    """Convert action indices to human-readable descriptions for a chunk of actions."""
    def format_actions(self, actions: Dict, start_idx: int, end_idx: int) -> Dict:
        chunk_actions = {}

        # Extract action slice
        for key in actions:
            if key == 'camera':
                chunk_actions[key] = actions[key][start_idx:end_idx].tolist()
            else:
                chunk_actions[key] = actions[key][start_idx:end_idx].tolist()

        # Create human-readable descriptions
        descriptions = []
        for i in range(len(chunk_actions['forward'])):
            step_actions = []

            # Movement actions
            if chunk_actions['forward'][i]:
                step_actions.append("forward")
            if chunk_actions['back'][i]:
                step_actions.append("back")
            if chunk_actions['left'][i]:
                step_actions.append("left")
            if chunk_actions['right'][i]:
                step_actions.append("right")

            # Discrete actions
            if chunk_actions['jump'][i]:
                step_actions.append("jump")
            if chunk_actions['sneak'][i]:
                step_actions.append("sneak")
            if chunk_actions['sprint'][i]:
                step_actions.append("sprint")
            if chunk_actions['attack'][i]:
                step_actions.append("attack")

            # Camera movement
            camera_x, camera_y = chunk_actions['camera'][i]
            if abs(camera_x) > 0.1 or abs(camera_y) > 0.1:
                step_actions.append(f"camera({camera_x:.1f}, {camera_y:.1f})")

            # Reward
            if chunk_actions['reward'][i] > 0:
                step_actions.append(f"reward={chunk_actions['reward'][i]:.1f}")

            # Create description
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
        episode_data = self.load_episode(episode_dir)
        if episode_data is None:
            return 0

        frames, actions = episode_data

        max_frames = len(frames)
        num_chunks = max_frames // self.chunk_size

        if num_chunks == 0:
            logger.warning(f"Episode {episode_dir.name} too short for chunks")
            return 0

        episode_output_dir = self.output_dir / episode_dir.name
        episode_output_dir.mkdir(exist_ok=True)

        chunks_created = 0
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * self.chunk_size
            end_frame = start_frame + self.chunk_size

            # Extract frame chunk
            frame_chunk = frames[start_frame:end_frame]

            # Extract and format action chunk
            action_chunk = self.format_actions(
                actions, start_frame, end_frame
            )

            # Save chunk
            chunk_dir = episode_output_dir / f"chunk_{chunk_idx:03d}"
            chunk_dir.mkdir(exist_ok=True)

            # Save frames as video or individual images
            self._save_chunk_frames(frame_chunk, chunk_dir)

            # Save action data
            self._save_chunk_actions(action_chunk, chunk_dir)

            # Save metadata
            metadata = {
                'episode': episode_dir.name,
                'chunk_index': chunk_idx,
                'frame_range': [start_frame, end_frame],
                'chunk_size': self.chunk_size,
                'action_summary': action_chunk['summary']
            }

            with open(chunk_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            chunks_created += 1

        logger.info(f"Created {chunks_created} chunks for episode {episode_dir.name}")
        return chunks_created

    def _save_chunk_frames(self, frames: np.ndarray, chunk_dir: Path):
        frames_dir = chunk_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(frames):
            frame_path = frames_dir / f"frame_{i:02d}.png"
            # Convert RGB back to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), frame_bgr)

        # Save as MP4 video
        self._save_chunk_video(frames, chunk_dir / "chunk.mp4")

        # Also save as numpy array for fast loading
        np.save(chunk_dir / "frames.npy", frames)

    def _save_chunk_video(self, frames: np.ndarray, video_path: Path):
        """Save frame sequence as MP4 video."""
        height, width = frames.shape[1:3]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0  # Match original video FPS
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    def _save_chunk_actions(self, action_chunk: Dict, chunk_dir: Path):
        # Save human-readable descriptions
        with open(chunk_dir / "actions.txt", 'w') as f:
            for i, desc in enumerate(action_chunk['descriptions']):
                f.write(f"Step {i:2d}: {desc}\n")

        # Save raw action data as JSON
        with open(chunk_dir / "actions.json", 'w') as f:
            json.dump(action_chunk, f, indent=2)

        # Save raw actions as numpy for fast loading
        np.save(chunk_dir / "actions.npy", action_chunk['raw_actions'])

    def process_all_episodes(self):
        episode_dirs = self.get_episode_directories()
        total_chunks = 0

        for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
            try:
                chunks_created = self.process_episode(episode_dir)
                total_chunks += chunks_created
            except Exception as e:
                logger.error(f"Error processing {episode_dir}: {e}")
                continue

        logger.info(f"Processing complete! Created {total_chunks} total chunks")

        # Save dataset summary
        summary = {
            'total_episodes': len(episode_dirs),
            'total_chunks': total_chunks,
            'chunk_size': self.chunk_size,
            'output_directory': str(self.output_dir)
        }

        with open(self.output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Process MineRL dataset into chunked format")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".data/MineRLTreechop-v0",
        help="Path to MineRL dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="chunked_dataset",
        help="Output directory for processed chunks"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Number of frames per chunk"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Limit number of episodes to process"
    )

    args = parser.parse_args()

    # Initialize processor
    processor = MineRLProcessor(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        chunk_size=args.chunk_size
    )

    # Process episodes
    if args.episodes:
        episode_dirs = processor.get_episode_dirs()[:args.episodes]
        total_chunks = 0
        for episode_dir in tqdm(episode_dirs, desc=f"Processing {args.episodes} episodes"):
            try:
                chunks_created = processor.process_episode(episode_dir)
                total_chunks += chunks_created
            except Exception as e:
                logger.error(f"Error processing {episode_dir}: {e}")
                continue
        logger.info(f"Created {total_chunks} chunks from {args.episodes} episodes")
    else:
        processor.process_all_episodes()


if __name__ == "__main__":
    main()
