"""
Multimodal Embedding Agent for MineRL

This agent uses fused multimodal embeddings combining:
1. MineCLIP video embeddings from 16-frame observations
2. Qwen2.5-VL generated text descriptions
3. MineCLIP text embeddings from descriptions
4. Fused embedding = average of video + text embeddings

Usage:
    python Agent/agent_multimodal.py

Environment Variables:
    USE_OPENAI_LLM: Set to "true" to use OpenAI instead of local model
    OPENAI_API_KEY: Required if USE_OPENAI_LLM is true
    USE_RAG: Set to "true" to enable episodic memory (uses fused embeddings)
    RAG_CONFIG: Set to "v4" for fused multimodal RAG
"""

from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import subprocess
import time
import requests
import base64
from PIL import Image
import io
import atexit
import json
import os
import re
import sys
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from dotenv import load_dotenv
import numpy as np
import cv2
import torch

# Load environment variables from .env file
load_dotenv()

# Check which LLM to use
USE_OPENAI_LLM = os.environ.get("USE_OPENAI_LLM", "false").lower() == "true"

# Import model setup utilities
from utils.constants import (
    OPENAI_SYSTEM_PROMPT,
    LOCAL_SYSTEM_PROMPT,
    MINERL_ACTION_TEMPLATE,
    PITCH_MIN,
    PITCH_MAX,
)

if USE_OPENAI_LLM:
    from utils.openai_model import setup_openai_agent, ensure_openai_api_key
    ensure_openai_api_key()
else:
    from utils.local_model import setup_local_agent


# =============================================================================
# Fused Embedding Generator
# =============================================================================

class FusedEmbeddingGenerator:
    """
    Generates fused multimodal embeddings by combining:
    - MineCLIP video embeddings (512-dim)
    - MineCLIP text embeddings from VLM-generated descriptions (512-dim)

    The fused embedding is the average of both: (video_emb + text_emb) / 2
    """

    def __init__(
        self,
        mineclip_checkpoint: str = ".ckpts/attn.pth",
        qwen_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto"
    ):
        """Initialize the fused embedding generator.

        Args:
            mineclip_checkpoint: Path to MineCLIP checkpoint file
            qwen_model_id: HuggingFace model ID for Qwen VLM
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.mineclip_checkpoint = mineclip_checkpoint
        self.qwen_model_id = qwen_model_id
        self.target_size = (160, 256)  # MineCLIP resolution

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"FusedEmbeddingGenerator using device: {self.device}")

        # Models (loaded lazily)
        self.mineclip_model = None
        self.qwen_model = None
        self.qwen_processor = None

    def _load_mineclip(self):
        """Load MineCLIP model for video and text encoding."""
        if self.mineclip_model is not None:
            return

        print("Loading MineCLIP model...")

        # Add parent directory to path for MineCLIP imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Patch VideoRewardBase before importing MineCLIP
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

        self.mineclip_model = MineCLIP(
            arch="vit_base_p16_fz.v2.t2",
            resolution=(160, 256),
            pool_type="attn.d2.nh8.glusw",
            image_feature_dim=512,
            mlp_adapter_spec="v0-2.t0",
            hidden_dim=512,
        ).to(self.device)

        self.mineclip_model.load_ckpt(self.mineclip_checkpoint, strict=False)
        self.mineclip_model.eval()

        print("MineCLIP model loaded successfully")

    def _load_qwen(self):
        """Load Qwen VLM for description generation."""
        if self.qwen_model is not None:
            return

        print(f"Loading Qwen VLM: {self.qwen_model_id}")

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "Please install transformers with: pip install transformers>=4.37.0"
            )

        self.qwen_processor = AutoProcessor.from_pretrained(self.qwen_model_id)

        # Determine dtype based on device
        if self.device.type == "cuda":
            torch_dtype = torch.float16
            device_map = "auto"
        else:
            torch_dtype = torch.float32
            device_map = str(self.device)

        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.qwen_model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        print("Qwen VLM loaded successfully")

    def _b64_to_numpy(self, b64_string: str) -> np.ndarray:
        """Convert a base64-encoded PNG/JPEG to a NumPy RGB image."""
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _load_frames_from_b64_list(self, b64_list: list) -> np.ndarray:
        """Convert a list of base64-encoded frames into a stacked NumPy array."""
        frames = []
        for b64_string in b64_list:
            frame = self._b64_to_numpy(b64_string)
            frames.append(frame)
        return np.stack(frames, axis=0)

    def _resize_frames(self, frames: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize frames to target resolution for MineCLIP."""
        resized = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (target_size[1], target_size[0]))
            resized.append(resized_frame)
        return np.array(resized)

    def _sample_frames(self, frames: np.ndarray, num_frames: int = 8) -> list:
        """Sample frames evenly from the window for VLM input.

        Args:
            frames: NumPy array of frames [N, H, W, 3]
            num_frames: Number of frames to sample (default 8)

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

    @torch.no_grad()
    def _encode_video(self, frames: np.ndarray) -> np.ndarray:
        """Encode 16 frames using MineCLIP video encoder.

        Args:
            frames: NumPy array of frames [16, H, W, 3]

        Returns:
            512-dimensional embedding vector
        """
        self._load_mineclip()

        frames_resized = self._resize_frames(frames, self.target_size)

        # [16, H, W, C] -> [1, 16, C, H, W]
        frames_tensor = (
            torch.from_numpy(frames_resized)
            .permute(0, 3, 1, 2)      # [16, 3, 160, 256]
            .unsqueeze(0)             # [1, 16, 3, 160, 256]
            .to(self.device)
        )

        torch_embedding = self.mineclip_model.encode_video(frames_tensor)
        embedding = torch_embedding.cpu().detach().numpy()

        return embedding[0]  # Shape: (512,)

    @torch.no_grad()
    def _generate_description(self, frames: np.ndarray) -> str:
        """Generate a description using Qwen VLM.

        Args:
            frames: NumPy array of frames [16, H, W, 3]

        Returns:
            Generated description string
        """
        self._load_qwen()

        # Sample 8 frames from the window
        images = self._sample_frames(frames, num_frames=8)

        # Build the prompt
        prompt = """You are analyzing a short video clip from Minecraft. The agent is trying to collect wood from trees.

Look at these frames from the video and describe what the agent sees in ONE short sentence.

Focus on:
- What objects are visible (trees, grass, dirt, sky, wood blocks)
- The agent's apparent position and view direction
- Any actions that seem to be happening

Respond with ONLY a single descriptive sentence, nothing else."""

        # Create message with images
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Process inputs
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.qwen_processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = inputs.to(self.qwen_model.device)

        # Generate
        generated_ids = self.qwen_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self.qwen_processor.tokenizer.pad_token_id
        )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output = self.qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output.strip()

    @torch.no_grad()
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text description using MineCLIP text encoder.

        Args:
            text: Description text

        Returns:
            512-dimensional embedding vector
        """
        self._load_mineclip()

        # MineCLIP's encode_text handles tokenization internally
        embedding = self.mineclip_model.clip_model.encode_text([text])

        return embedding[0].cpu().numpy()  # Shape: (512,)

    def generate_fused_embedding(self, frames_b64: list) -> tuple:
        """Generate fused embedding from 16 base64-encoded frames.

        Args:
            frames_b64: List of 16 base64-encoded frame strings

        Returns:
            Tuple of (fused_embedding, description):
                - fused_embedding: 512-dimensional numpy array
                - description: Generated text description
        """
        # Decode frames
        frames = self._load_frames_from_b64_list(frames_b64)

        # Step 1: Generate video embedding
        print("  Generating video embedding...")
        video_embedding = self._encode_video(frames)

        # Step 2: Generate description
        print("  Generating description...")
        description = self._generate_description(frames)
        print(f"  Description: {description}")

        # Step 3: Generate text embedding
        print("  Generating text embedding...")
        text_embedding = self._encode_text(description)

        # Step 4: Fuse embeddings (average)
        fused_embedding = (video_embedding + text_embedding) / 2.0

        return fused_embedding, description


# =============================================================================
# Configuration from environment variables
# =============================================================================

# OpenAI LLM configuration
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))

# Local LLM configuration
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
LOCAL_MODEL_DEVICE = os.environ.get("LOCAL_MODEL_DEVICE", "auto")
LOCAL_MODEL_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_MODEL_MAX_NEW_TOKENS", "256"))
LOCAL_MODEL_TEMPERATURE = float(os.environ.get("LOCAL_MODEL_TEMPERATURE", "0.2"))
LOCAL_MODEL_DTYPE = os.environ.get("LOCAL_MODEL_DTYPE", "float16")

# Agent configuration
USE_RAG = os.environ.get("USE_RAG", "false").lower() == "true"
RAG_CONFIG = os.environ.get("RAG_CONFIG", "v4")  # Default to v4 for fused embeddings
RENDER = os.environ.get("RENDER", "false").lower() == "true"
TEST_RUNS = int(os.environ.get("TEST_RUNS", "5"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "500"))
USE_MAX_FRAMES = os.environ.get("USE_MAX_FRAMES", "true").lower() == "true"
RESTART_THRESHOLD = int(os.environ.get("RESTART_THRESHOLD", "200"))  # Restart if no wood within this many frames

# Remote MineRL server configuration
USE_REMOTE_SERVER = os.environ.get("USE_REMOTE_SERVER", "false").lower() == "true"
SERVER_URL = os.environ.get("MINERL_SERVER_URL", "http://127.0.0.1:5001")
ENV_ID = None

# Remote model server configuration (for Qwen VL + MineCLIP)
USE_REMOTE_MODEL_SERVER = os.environ.get("USE_REMOTE_MODEL_SERVER", "false").lower() == "true"
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:8080")
MODEL_SERVER_TIMEOUT = int(os.environ.get("MODEL_SERVER_TIMEOUT", "120"))

print("Configuration:")
print(f"  Use OpenAI LLM: {USE_OPENAI_LLM}")
if USE_OPENAI_LLM:
    print(f"  OpenAI Model: {OPENAI_MODEL_NAME}")
    print(f"  OpenAI Temperature: {OPENAI_TEMPERATURE}")
else:
    print(f"  Local Model: {LOCAL_MODEL_NAME}")
    print(f"  Device: {LOCAL_MODEL_DEVICE}")
    print(f"  Max New Tokens: {LOCAL_MODEL_MAX_NEW_TOKENS}")
    print(f"  Temperature: {LOCAL_MODEL_TEMPERATURE}")
print(f"  Use RAG: {USE_RAG}")
print(f"  RAG Config: {RAG_CONFIG}")
print(f"  Render: {RENDER}")
print(f"  Test Runs: {TEST_RUNS}")
print(f"  Max Frames: {MAX_FRAMES}")
print(f"  Restart Threshold: {RESTART_THRESHOLD}")
print(f"  Use Remote MineRL Server: {USE_REMOTE_SERVER}")
print(f"  MineRL Server URL: {SERVER_URL}")
print(f"  Use Remote Model Server: {USE_REMOTE_MODEL_SERVER}")
if USE_REMOTE_MODEL_SERVER:
    print(f"  Model Server URL: {MODEL_SERVER_URL}")
    print(f"  Model Server Timeout: {MODEL_SERVER_TIMEOUT}s")


# =============================================================================
# State tracking
# =============================================================================

ACTION_HISTORY = deque(maxlen=5)
FRAME_HISTORY = deque(maxlen=16)

CAMERA_STATE = {"pitch": 0.0, "yaw": 0.0}

# Note: The server already handles attack repetition (15 ticks per attack action)
# so we don't need agent-side sustained attacks. The agent just needs to
# send attack once and the server will hold it long enough to break blocks.

# For rendering the frames
_RENDER = {
    "fig": None,
    "ax": None,
    "im": None,
}

# Initialize RAG system with fused embeddings
rag = None
fused_embedding_generator = None

if USE_RAG:
    try:
        from RAG_server_v3 import RAGFusedEmbedding
        rag = RAGFusedEmbedding()
        print("RAG v4 (fused multimodal) initialized successfully")
    except ImportError as e:
        print(f"Warning: RAG_server_v3 not available: {e}")
        # Fallback to standard RAG if available
        try:
            from RAG_server import create_rag
            rag = create_rag(RAG_CONFIG)
            # Also initialize fused embedding generator for queries
            fused_embedding_generator = FusedEmbeddingGenerator()
            print(f"Using standard RAG with config: {RAG_CONFIG}")
        except ImportError:
            print("Warning: RAG requested but not available")

# Load example images for training examples
try:
    with open('example_images.json', 'r') as f:
        example_images = json.load(f)
    EXAMPLE_IMAGES_LOADED = True
except FileNotFoundError:
    try:
        with open('Agent/example_images.json', 'r') as f:
            example_images = json.load(f)
        EXAMPLE_IMAGES_LOADED = True
    except FileNotFoundError:
        print("Warning: example_images.json not found, using text-only examples")
        example_images = {}
        EXAMPLE_IMAGES_LOADED = False

# Create base64 data URLs for example images
example_image_1 = "data:image/png;base64," + example_images.get("example_1", "") if EXAMPLE_IMAGES_LOADED else None
example_image_2 = "data:image/png;base64," + example_images.get("example_2", "") if EXAMPLE_IMAGES_LOADED else None
example_image_3 = "data:image/png;base64," + example_images.get("example_3", "") if EXAMPLE_IMAGES_LOADED else None
example_image_4 = "data:image/png;base64," + example_images.get("example_4", "") if EXAMPLE_IMAGES_LOADED else None

# Example actions and explanations
example_action_1 = {"forward": 1}
example_action_2 = {"attack": 1}
example_action_3 = {"camera": [0, -30]}
example_action_4 = {"camera": [30, 0]}

explanation_1 = "The agent saw the tree (object in white) in the distance and started moving towards it."
explanation_2 = "The agent is now at the tree. It continuously attacks the tree until the block is gone. Once the block is broken the reward is given."
explanation_3 = "The agent can only see dirt. There is nothing useful here and it should not be attacked. The agent should look around to find trees."
explanation_4 = "The agent just destroyed wood. To collect more the viewpoint needs to be changed. The agent can look up or down."


# =============================================================================
# System prompt for the agent
# =============================================================================

def build_openai_system_prompt():
    """Build the system prompt for OpenAI models with detailed behavioral examples including images."""
    base_prompt = OPENAI_SYSTEM_PROMPT

    if EXAMPLE_IMAGES_LOADED:
        training_examples = f"""

Training examples:

EXAMPLE 1:
Input image: [{example_image_1}]
Taken action: {example_action_1}
The reason for the action: {explanation_1}

EXAMPLE 2:
Input image: [{example_image_2}]
Taken action: {example_action_2}
The reason for the action: {explanation_2}

EXAMPLE 3:
Input image: [{example_image_3}]
Taken action: {example_action_3}
The reason for the action: {explanation_3}

EXAMPLE 4:
Input image: [{example_image_4}]
Taken action: {example_action_4}
The reason for the action: {explanation_4}

Key visual cues for trees:
- Trees have brown/tan vertical trunks
- Oak logs appear as brown/beige blocks
- Trees are taller than the ground and stand out against the sky
- When close enough to attack, the trunk should be in the CENTER of your view
"""
    else:
        training_examples = """

Detailed behavioral examples:

EXAMPLE 1 - Tree visible in distance:
- What you see: A tree trunk (brown/white vertical structure) visible ahead
- Best action: {"forward": 1} - Move towards the tree
- Why: The agent saw the tree in the distance and started moving towards it.

EXAMPLE 2 - Close to tree trunk:
- What you see: Tree trunk fills center of view, very close
- Best action: {"attack": 1} - Attack to break the wood block
- Why: The agent is now at the tree. Attack continuously until the block breaks and reward is given.

EXAMPLE 3 - No trees visible (dirt/grass only):
- What you see: Only ground (dirt, grass) with no trees in view
- Best action: {"camera": [0, -30]} or {"camera": [0, 30]} - Look around
- Why: The agent can only see dirt. There is nothing useful here. Look around to find trees.

EXAMPLE 4 - Just collected wood, need to find more:
- What you see: Tree partially destroyed or wood block just collected
- Best action: {"camera": [15, 0]} or {"camera": [-15, 0]} - Adjust view up/down
- Why: The agent just destroyed wood. To collect more, adjust the viewpoint to see remaining tree or find new trees.

Key visual cues for trees:
- Trees have brown/tan vertical trunks
- Oak logs appear as brown/beige blocks
- Trees are taller than the ground and stand out against the sky
- When close enough to attack, the trunk should be in the CENTER of your view
"""
    return base_prompt + training_examples

# Select the appropriate system prompt based on LLM type
if USE_OPENAI_LLM:
    system_msg = SystemMessage(content=build_openai_system_prompt())
else:
    system_msg = SystemMessage(content=LOCAL_SYSTEM_PROMPT)

container_started = False


# =============================================================================
# Utility functions
# =============================================================================

def log_result(wood):
    """Save the results of a run."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wood_collected": wood
    }

    path = f'Agent/Results/{RAG_CONFIG}.jsonl'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def process_image(obs_b64):
    """Convert base64 to PIL Image."""
    obs_bytes = base64.b64decode(obs_b64)
    obs = Image.open(io.BytesIO(obs_bytes))
    return obs


def init_renderer(first_obs_b64: str):
    """Create the matplotlib window ONCE."""
    frame = process_image(first_obs_b64)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.set_axis_off()

    im = ax.imshow(frame, interpolation="nearest")
    fig.tight_layout(pad=0)

    _RENDER["fig"] = fig
    _RENDER["ax"] = ax
    _RENDER["im"] = im

    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()


def show_obs(obs_b64: str):
    """Update the existing window instead of clearing/recreating."""
    if _RENDER["fig"] is None:
        init_renderer(obs_b64)
        return

    frame = process_image(obs_b64)
    _RENDER["im"].set_data(frame)

    fig = _RENDER["fig"]
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


def exit_cleanup():
    """Clean up resources on exit."""
    global container_started, ENV_ID

    if USE_REMOTE_SERVER:
        if ENV_ID:
            try:
                requests.post(
                    f"{SERVER_URL}/destroy",
                    json={"env_id": ENV_ID},
                    timeout=5
                )
                print(f"Destroyed remote environment session: {ENV_ID}")
            except Exception as e:
                print(f"Failed to destroy remote environment: {e}")
    else:
        if container_started:
            try:
                subprocess.run(
                    ["docker", "compose", "stop", "minerl-env"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                print("Stopping docker-compose service minerl-env")
            except Exception as e:
                print(f"Failed to stop docker-compose service: {e}")


atexit.register(exit_cleanup)


# =============================================================================
# Environment management
# =============================================================================

def create_env():
    """Start the environment (local Docker or remote server)."""
    global container_started, ENV_ID

    if USE_REMOTE_SERVER:
        print(f"Connecting to remote MineRL server at {SERVER_URL}...")

        try:
            resp = requests.post(
                f"{SERVER_URL}/create",
                json={"env_type": "MineRLTreechop-v0"},
            )

        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout connecting to MineRL server at {SERVER_URL}. Is the server running?")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Could not connect to MineRL server at {SERVER_URL}. Is the server running?\nError: {e}")

        if resp.status_code != 200:
            raise RuntimeError(f'Failed to create remote environment: {resp.status_code} - {resp.text}')

        data = resp.json()
        ENV_ID = data['env_id']

        time.sleep(60)

        print(f"Created remote environment session: {ENV_ID}")

        return data['obs']
    else:
        subprocess.run(
            ["docker", "compose", "up", "-d", "minerl-env"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        container_started = True
        print("Started environment server using docker-compose (service: minerl-env). Waiting for setup.")
        time.sleep(20)

        print('Setting up environment...')
        print('LONG WAIT')
        resp = requests.get(url=f'{SERVER_URL}/start')

        if resp.status_code != 200:
            raise RuntimeError(f'Status code error: {resp.status_code}')

        return resp.json()['obs']


def reset_env():
    """Reset the environment for different runs."""
    if USE_REMOTE_SERVER:
        resp = requests.post(
            f"{SERVER_URL}/reset",
            json={"env_id": ENV_ID}
        )
    else:
        resp = requests.post(url=f'{SERVER_URL}/reset')

    if resp.status_code != 200:
        raise RuntimeError(f'Status code error: {resp.status_code}')

    return resp.json()['obs']


def submit_action(action_dict: dict):
    """Submit actions to the server."""
    action_body = MINERL_ACTION_TEMPLATE.copy()

    global CAMERA_STATE

    for action, value in action_dict.items():
        if action == "camera":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("Camera action requires value=[pitch_delta, yaw_delta]")

            pitch_delta, yaw_delta = float(value[0]), float(value[1])

            new_pitch = CAMERA_STATE["pitch"] + pitch_delta
            new_pitch = max(min(new_pitch, PITCH_MAX), PITCH_MIN)
            CAMERA_STATE["pitch"] = new_pitch
            CAMERA_STATE["yaw"] += yaw_delta

            actual_pitch_delta = new_pitch - (CAMERA_STATE["pitch"] - pitch_delta)
            action_body["camera"] = [actual_pitch_delta, yaw_delta]
        else:
            if action not in action_body:
                print(f"Warning: Ignoring unknown action name: {action}")
                continue
            action_body[action] = int(value)

    body = {"actions": action_body}
    if USE_REMOTE_SERVER:
        body["env_id"] = ENV_ID

    body["sustained_attack_count"] = 30

    resp = requests.post(
        url=f"{SERVER_URL}/action",
        json=body,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Status code error: {resp.status_code}")

    data = resp.json()
    obs = data["obs"]
    reward = data["reward"]
    done = data["done"]

    return obs, reward, done, action_body


# =============================================================================
# Tool definition for LLM
# =============================================================================

@tool
def step_env(action: dict) -> dict:
    """Step the MineRL environment by executing a combined action.

    Args:
        action: Dictionary specifying controls to activate.
            Valid keys: "forward", "back", "left", "right", "jump", "attack", "camera"
            Movement/attack: 0 or 1
            Camera: [pitch_delta, yaw_delta]

    Returns:
        Dictionary with obs, reward, and done status.
    """
    obs, reward, done, action_body = submit_action(action)

    print(f"Action: {action}")

    ACTION_HISTORY.append({
        "action": action,
        "action_body": action_body,
        "reward": reward,
    })

    return {
        "obs": obs,
        "reward": reward,
        "done": done,
    }


# =============================================================================
# Action parsing
# =============================================================================

def parse_action_from_text(text: str) -> dict | None:
    """
    Attempt to parse an action dict from model output text.
    Handles various formats that local models might output.
    """
    json_pattern = r'\{[^{}]*(?:"forward"|"back"|"left"|"right"|"jump"|"attack"|"camera")[^{}]*\}'
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            action = json.loads(match)
            valid_keys = {"forward", "back", "left", "right", "jump", "attack", "camera"}
            if any(k in valid_keys for k in action.keys()):
                return action
        except json.JSONDecodeError:
            continue

    action_pattern = r'action\s*=\s*(\{[^{}]+\})'
    match = re.search(action_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    tool_call_pattern = r'step_env\s*\(\s*action\s*=\s*(\{[^{}]+\})\s*\)'
    match = re.search(tool_call_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# Message formatting
# =============================================================================

def format_prev_actions():
    """Format the previous actions list."""
    if ACTION_HISTORY:
        history_lines = [
            f"{i+1}. action={entry['action']}, reward={entry['reward']}"
            for i, entry in enumerate(list(ACTION_HISTORY))
        ]
        context = "Last actions taken:\n" + "\n".join(history_lines)
    else:
        context = "No previous actions yet."

    return context


def format_user_msg(context: str, memory_str: str | None = None, obs: str | None = None):
    """
    Format user message.

    For OpenAI models: includes vision (image) and uses tool calling.
    For local models: text-only, expects JSON action response.
    """
    prev_actions = format_prev_actions()
    goal = 'Goal: Collect as much wood as possible.'

    memory_block = memory_str or "No relevant episodic memory was retrieved for this state."

    camera_info = (
        f"Estimated camera pitch: {CAMERA_STATE['pitch']:.1f} degrees "
        "(0 = horizon; positive = down, negative = up).\n"
        "Try to keep pitch between about -20 and +20 degrees unless you are "
        "deliberately looking up or down briefly.\n"
    )

    if USE_OPENAI_LLM and obs is not None:
        user_msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "You see the current Minecraft frame below.\n\n"
                        f"Extra context:\n{context}\n\n"
                        f"Camera state:\n{camera_info}\n"
                        f"Previous actions:\n{prev_actions}\n\n"
                        f"Episodic memory (similar past situation):\n{memory_block}\n\n"
                        f"Goal: {goal}\n\n"
                        "Remember: You only get reward when you collect WOOD LOGS from trees. "
                        "If there is no reward, you did not hit the wood. "
                        "Decide the next action and call the `step_env` tool."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{obs}"
                    },
                },
            ]
        )
    else:
        user_msg = HumanMessage(
            content=(
                "You are in a Minecraft world. Based on your memory and previous actions, "
                "decide the next action to collect wood.\n\n"
                f"Extra context:\n{context}\n\n"
                f"Camera state:\n{camera_info}\n"
                f"Previous actions:\n{prev_actions}\n\n"
                f"Episodic memory (similar past situation):\n{memory_block}\n\n"
                f"Goal: {goal}\n\n"
                "Respond with ONLY a JSON action dict like: {\"forward\": 1} or {\"camera\": [0, 15]}\n"
            )
        )

    return user_msg


# =============================================================================
# Agent episode runner
# =============================================================================

# Note: Sustained attack functions removed - server handles attack repetition automatically


def check_rag_suggests_attack(memory_str: str) -> bool:
    """Check if RAG memory suggests attacking."""
    attack_keywords = ["attack", "hitting", "breaking", "mining", "chopping"]
    memory_lower = memory_str.lower()
    return any(keyword in memory_lower for keyword in attack_keywords)


def check_looking_down(description: str) -> bool:
    """Check if the description indicates agent is looking down at ground."""
    desc_lower = description.lower()
    # Looking down at ground means we're not looking at tree trunk
    looking_down = "looking down" in desc_lower
    seeing_ground = ("ground" in desc_lower or "dirt" in desc_lower) and "tree" not in desc_lower.split("looking")[0] if "looking" in desc_lower else False
    return looking_down or seeing_ground


def needs_pitch_correction() -> bool:
    """Check if camera pitch needs correction (looking too far down)."""
    # If pitch is positive (looking down), we should look up
    return CAMERA_STATE["pitch"] > 10.0


def get_pitch_correction_action() -> dict:
    """Get action to correct camera pitch to look more level/up at trees."""
    # Look up by adjusting pitch negatively
    correction = min(-15.0, -CAMERA_STATE["pitch"])  # Look up
    print(f"  [Pitch Correction] Current pitch: {CAMERA_STATE['pitch']:.1f}, adjusting by {correction:.1f}")
    return {"camera": [correction, 0]}


def run_agent_episode(agent, obs):
    """Run a single agent episode step.

    Note: The server already handles attack repetition (15 ticks per attack),
    so we don't need agent-side sustained attacks. One attack action is enough
    to break a block.
    """
    reward = 0
    done = False

    # Query episodic memory using fused embeddings if available
    memory_str = "No episodic memory yet (not enough frames)."
    rag_suggests_attack = False
    current_description = ""
    if USE_RAG and rag and len(FRAME_HISTORY) > 0:
        # Pad frame buffer to 16 frames by duplicating first frame if needed
        frames_list = list(FRAME_HISTORY)
        if len(frames_list) < 16:
            first_frame = frames_list[0]
            padding_needed = 16 - len(frames_list)
            frames_list = [first_frame] * padding_needed + frames_list
            print(f"\nQuerying episodic memory with fused embedding (padded {padding_needed} frames)...")
        else:
            print("\nQuerying episodic memory with fused embedding...")
        memory_str = rag.get_action(frames_list)
        # Extract current description from memory string
        if "Current observation:" in memory_str:
            try:
                current_description = memory_str.split("Current observation:")[1].split("\n")[0].strip()
            except (IndexError, ValueError):
                pass
        rag_suggests_attack = check_rag_suggests_attack(memory_str)

    context = ""
    user_msg = format_user_msg(context, memory_str=memory_str, obs=obs)

    # Invoke the model with retry logic for rate limiting
    ai_msg = None
    max_retries = 5
    retry_delay = 6

    for attempt in range(max_retries):
        try:
            ai_msg = agent.invoke([system_msg, user_msg])
            break
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in str(e):
                print(f"Hit rate limit. Waiting {retry_delay}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                print(f"Error invoking model: {e}")
                raise

    if ai_msg is None:
        print("Warning: Failed to get response from model after retries.")
        # Fallback: if RAG suggests attack, do attack (but check pitch first)
        if rag_suggests_attack:
            looking_down = check_looking_down(current_description) if current_description else False
            pitch_too_low = needs_pitch_correction()

            if looking_down or pitch_too_low:
                print("  Fallback: RAG suggests attack, but looking down. Correcting pitch first.")
                correction_action = get_pitch_correction_action()
                result = step_env.invoke({"action": correction_action})
                FRAME_HISTORY.append(result["obs"])
                return result["obs"], result["reward"], result["done"]
            else:
                print("  Fallback: RAG suggests attack, executing attack.")
                result = step_env.invoke({"action": {"attack": 1}})
                FRAME_HISTORY.append(result["obs"])
                return result["obs"], result["reward"], result["done"]
        FRAME_HISTORY.append(obs)
        return obs, 0, False

    action_executed = False

    # Try standard tool_calls first
    if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
            if tool_name == "step_env":
                raw_args = tool_call.get("args") if isinstance(tool_call, dict) else tool_call.args

                if not isinstance(raw_args, dict):
                    print(f"Warning: Tool call args is not a dict: {raw_args}. Skipping action (no-op).")
                    FRAME_HISTORY.append(obs)
                    return obs, 0, False
                if "action" not in raw_args:
                    raw_args = {"action": raw_args}

                action_value = raw_args.get("action")
                if not isinstance(action_value, dict):
                    print(f"Warning: action is not a dict: {action_value}. Skipping action (no-op).")
                    FRAME_HISTORY.append(obs)
                    return obs, 0, False

                # Check if attack action needs pitch correction first
                if action_value.get("attack", 0) == 1:
                    looking_down = check_looking_down(current_description) if current_description else False
                    pitch_too_low = needs_pitch_correction()

                    if looking_down or pitch_too_low:
                        print("  [Attack Prep] Looking down detected. Correcting pitch before attacking...")
                        correction_action = get_pitch_correction_action()
                        result = step_env.invoke({"action": correction_action})
                        FRAME_HISTORY.append(result["obs"])
                        return result["obs"], result["reward"], result["done"]

                try:
                    result = step_env.invoke(raw_args)
                    obs = result["obs"]
                    reward = result["reward"]
                    done = result["done"]
                    action_executed = True
                except Exception as e:
                    print(f"Warning: Tool invocation failed: {e}. Skipping action (no-op).")
                    FRAME_HISTORY.append(obs)
                    return obs, 0, False

    # Fallback for local models: parse action from text output
    if not action_executed and not USE_OPENAI_LLM:
        text_content = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
        parsed_action = parse_action_from_text(text_content)

        if parsed_action:
            print(f"Parsed action from text: {parsed_action}")

            # Check if attack action needs pitch correction first
            if parsed_action.get("attack", 0) == 1:
                looking_down = check_looking_down(current_description) if current_description else False
                pitch_too_low = needs_pitch_correction()

                if looking_down or pitch_too_low:
                    print("  [Attack Prep] Looking down detected. Correcting pitch before attacking...")
                    correction_action = get_pitch_correction_action()
                    result = step_env.invoke({"action": correction_action})
                    FRAME_HISTORY.append(result["obs"])
                    return result["obs"], result["reward"], result["done"]

            result = step_env.invoke({"action": parsed_action})
            obs = result["obs"]
            reward = result["reward"]
            done = result["done"]
            action_executed = True
        else:
            print(f"Warning: Could not parse action from model output: {text_content[:300]}...")
            # Fallback: if RAG suggests attack, do attack (but check pitch first)
            if rag_suggests_attack:
                looking_down = check_looking_down(current_description) if current_description else False
                pitch_too_low = needs_pitch_correction()

                if looking_down or pitch_too_low:
                    print("  Fallback: RAG suggests attack but looking down. Correcting pitch first.")
                    correction_action = get_pitch_correction_action()
                    result = step_env.invoke({"action": correction_action})
                    FRAME_HISTORY.append(result["obs"])
                    return result["obs"], result["reward"], result["done"]
                else:
                    print("  Fallback: RAG suggests attack, executing attack.")
                    result = step_env.invoke({"action": {"attack": 1}})
                    FRAME_HISTORY.append(result["obs"])
                    return result["obs"], result["reward"], result["done"]
            print("Skipping action (no-op).")
            FRAME_HISTORY.append(obs)
            return obs, 0, False
    elif not action_executed and USE_OPENAI_LLM:
        print("Warning: OpenAI model did not make a tool call.")
        # Fallback: if RAG suggests attack or we were recently attacking, do attack
        last_action_was_attack = (
            ACTION_HISTORY and
            ACTION_HISTORY[-1].get("action", {}).get("attack", 0) == 1
        )
        if rag_suggests_attack or last_action_was_attack:
            looking_down = check_looking_down(current_description) if current_description else False
            pitch_too_low = needs_pitch_correction()

            if looking_down or pitch_too_low:
                print("  Fallback: Attack suggested but looking down. Correcting pitch first.")
                correction_action = get_pitch_correction_action()
                result = step_env.invoke({"action": correction_action})
                FRAME_HISTORY.append(result["obs"])
                return result["obs"], result["reward"], result["done"]
            else:
                print("  Fallback: Executing attack based on RAG or previous action.")
                result = step_env.invoke({"action": {"attack": 1}})
                FRAME_HISTORY.append(result["obs"])
                return result["obs"], result["reward"], result["done"]
        print("Skipping action (no-op).")
        FRAME_HISTORY.append(obs)
        return obs, 0, False

    FRAME_HISTORY.append(obs)

    return obs, reward, done


# =============================================================================
# Main agent loop
# =============================================================================

def run_agent():
    """Main agent execution loop."""
    print("\n" + "="*50)
    print("Starting Multimodal Embedding Agent...")
    print("="*50 + "\n")

    # Setup the agent based on configuration
    if USE_OPENAI_LLM:
        print("Setting up OpenAI LLM agent...")
        agent = setup_openai_agent(tools=[step_env])
    else:
        print("Setting up local LLM agent...")
        agent = setup_local_agent()

    # Start the env
    print("\nConnecting to MineRL environment...")
    print(f"  Server URL: {SERVER_URL}")
    print(f"  Remote mode: {USE_REMOTE_SERVER}")
    obs = create_env()

    print("Environment connected successfully!")

    # Start rendering
    if RENDER:
        init_renderer(obs)

    done = False
    reward = 0
    wood_count = 0
    cur_frames = 0

    run_idx = 0
    while run_idx < TEST_RUNS:
        print(f"\n{'='*50}")
        print(f"Starting run {run_idx + 1}/{TEST_RUNS}")
        print(f"{'='*50}\n")

        done = False
        restart_run = False

        while not done:
            obs, reward, done = run_agent_episode(agent, obs)

            if reward != 0:
                wood_count += 1
                print(f'Wood count: {wood_count}')

            if RENDER:
                show_obs(obs)

            cur_frames += 1
            print(f'Frame {cur_frames}')

            # Check if no wood collected within the restart threshold
            if cur_frames >= RESTART_THRESHOLD and wood_count == 0:
                print(f"\n[RESTART] No wood collected within first {RESTART_THRESHOLD} frames. Restarting run in new environment...")
                restart_run = True
                break

            if USE_MAX_FRAMES and cur_frames >= MAX_FRAMES:
                cur_frames = 0
                break

        if restart_run:
            # Reset for a new attempt without counting this as a completed run
            wood_count = 0
            cur_frames = 0
            ACTION_HISTORY.clear()
            FRAME_HISTORY.clear()
            obs = reset_env()
            continue  # Retry the same run_idx

        log_result(wood_count)
        print(f"\nRun {run_idx + 1} complete. Wood collected: {wood_count}")

        wood_count = 0
        cur_frames = 0
        ACTION_HISTORY.clear()
        FRAME_HISTORY.clear()

        if run_idx < TEST_RUNS - 1:
            obs = reset_env()

        run_idx += 1

    print("\nAll runs complete!")


if __name__ == "__main__":
    run_agent()
