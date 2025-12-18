"""
6-Step Dataset Processing Pipeline

This module provides a complete pipeline for processing MineRL episodes into
a vectorDB-ready format with fused embeddings and LLM-derived actions.

Steps:
1. Sliding Window Chunking - Create overlapping windows from episodes
2. Video Embedding - Encode windows using MineCLIP video encoder
3. Generate Descriptions - Use Qwen VLM to describe each window
4. Text Embedding - Encode descriptions using MineCLIP text encoder
5. Generate Actions - Use Qwen VLM to predict next best actions
6. Export CSV - Create fused embeddings CSV for vectorDB
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_window_dirs(data_dir: Path) -> List[Path]:
    """Get all window directories from the dataset.

    Args:
        data_dir: Path to the dataset directory containing episode subdirs

    Returns:
        Sorted list of window directory paths
    """
    window_dirs = []
    data_dir = Path(data_dir)

    if not data_dir.exists():
        logger.warning(f"Dataset directory not found: {data_dir}")
        return window_dirs

    for episode_dir in data_dir.iterdir():
        if episode_dir.is_dir():
            for window_dir in episode_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    window_dirs.append(window_dir)

    return sorted(window_dirs)


def setup_mineclip_imports():
    """Setup sys.path for MineCLIP imports."""
    # Add parent directory to path for MineCLIP imports
    parent_dir = Path(__file__).resolve().parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))


def load_mineclip_model(checkpoint_path: str, device: torch.device):
    """Load MineCLIP model from checkpoint.

    Args:
        checkpoint_path: Path to the MineCLIP checkpoint file
        device: Device to load the model on

    Returns:
        Loaded MineCLIP model in eval mode
    """
    import torch.nn as nn

    # Patch VideoRewardBase before importing MineCLIP
    class VideoRewardBase(nn.Module):
        def __init__(self, *, image_encoder, temporal_encoder, reward_head):
            super().__init__()
            self.image_encoder = image_encoder
            self.temporal_encoder = temporal_encoder
            self.reward_head = reward_head

    setup_mineclip_imports()

    import MineCLIP.mineclip.base
    MineCLIP.mineclip.base.VideoRewardBase = VideoRewardBase

    from MineCLIP.mineclip import MineCLIP

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


# Export public API
__all__ = [
    "logger",
    "get_device",
    "get_window_dirs",
    "setup_mineclip_imports",
    "load_mineclip_model",
]
