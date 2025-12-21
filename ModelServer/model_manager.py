"""
Model Manager - Singleton for loading and managing Qwen VL and MineCLIP models.

Extracted and adapted from Agent/agent_multimodal.py FusedEmbeddingGenerator class.
"""

import os
import sys
import threading
from typing import Optional, Tuple
import base64

import numpy as np
import cv2
import torch
from PIL import Image


class ModelManager:
    """
    Singleton model manager for Qwen VL and MineCLIP models.

    Features:
    - Lazy loading: models are loaded on first use
    - Thread-safe model loading with locks
    - Auto-detect device: CUDA > MPS > CPU
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._mineclip_model = None
        self._qwen_model = None
        self._qwen_processor = None
        self._model_lock = threading.Lock()

        # Auto-detect device
        self.device = self._detect_device()
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Configuration
        self.mineclip_checkpoint = os.environ.get(
            "MINECLIP_CHECKPOINT",
            "/app/.ckpts/attn.pth"
        )
        self.qwen_model_id = os.environ.get(
            "QWEN_MODEL_ID",
            "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        self.target_size = (160, 256)  # MineCLIP resolution (H, W)

        print(f"ModelManager initialized on device: {self.device}")
        print(f"  MineCLIP checkpoint: {self.mineclip_checkpoint}")
        print(f"  Qwen model: {self.qwen_model_id}")

    def _detect_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_mineclip(self):
        """Load MineCLIP model for video and text encoding."""
        if self._mineclip_model is not None:
            return

        with self._model_lock:
            if self._mineclip_model is not None:
                return

            print("Loading MineCLIP model...")

            # Add MineCLIP to path
            mineclip_path = "/app/MineCLIP"
            if mineclip_path not in sys.path:
                sys.path.insert(0, mineclip_path)

            # Also try parent directory pattern
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

            self._mineclip_model = MineCLIP(
                arch="vit_base_p16_fz.v2.t2",
                resolution=(160, 256),
                pool_type="attn.d2.nh8.glusw",
                image_feature_dim=512,
                mlp_adapter_spec="v0-2.t0",
                hidden_dim=512,
            ).to(self.device)

            self._mineclip_model.load_ckpt(self.mineclip_checkpoint, strict=False)
            self._mineclip_model.eval()

            print("MineCLIP model loaded successfully")

    def _load_qwen(self):
        """Load Qwen VLM for description generation."""
        if self._qwen_model is not None:
            return

        with self._model_lock:
            if self._qwen_model is not None:
                return

            print(f"Loading Qwen VLM: {self.qwen_model_id}")

            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            self._qwen_processor = AutoProcessor.from_pretrained(self.qwen_model_id)

            # Determine dtype based on device
            if self.device.type == "cuda":
                torch_dtype = torch.float16
                device_map = "auto"
            else:
                torch_dtype = torch.float32
                device_map = str(self.device)

            self._qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        """Sample frames evenly from the window for VLM input."""
        total_frames = len(frames)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        sampled = []
        for idx in indices:
            img = Image.fromarray(frames[idx].astype(np.uint8))
            sampled.append(img)

        return sampled

    @torch.no_grad()
    def encode_video(self, frames_b64: list) -> np.ndarray:
        """
        Encode 16 frames using MineCLIP video encoder.

        Args:
            frames_b64: List of 16 base64-encoded frame strings

        Returns:
            512-dimensional embedding vector as numpy array
        """
        self._load_mineclip()

        frames = self._load_frames_from_b64_list(frames_b64)
        frames_resized = self._resize_frames(frames, self.target_size)

        # [16, H, W, C] -> [1, 16, C, H, W]
        frames_tensor = (
            torch.from_numpy(frames_resized)
            .permute(0, 3, 1, 2)      # [16, 3, 160, 256]
            .unsqueeze(0)             # [1, 16, 3, 160, 256]
            .to(self.device)
        )

        torch_embedding = self._mineclip_model.encode_video(frames_tensor)
        embedding = torch_embedding.cpu().detach().numpy()

        return embedding[0]  # Shape: (512,)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text description using MineCLIP text encoder.

        Args:
            text: Description text

        Returns:
            512-dimensional embedding vector as numpy array
        """
        self._load_mineclip()

        # MineCLIP's encode_text handles tokenization internally
        embedding = self._mineclip_model.clip_model.encode_text([text])

        return embedding[0].cpu().numpy()  # Shape: (512,)

    @torch.no_grad()
    def generate_description(self, frames_b64: list) -> str:
        """
        Generate a description using Qwen VLM.

        Args:
            frames_b64: List of 16 base64-encoded frame strings

        Returns:
            Generated description string
        """
        self._load_qwen()

        frames = self._load_frames_from_b64_list(frames_b64)

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
        text = self._qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._qwen_processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = inputs.to(self._qwen_model.device)

        # Generate
        generated_ids = self._qwen_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self._qwen_processor.tokenizer.pad_token_id
        )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output = self._qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output.strip()

    def generate_fused_embedding(
        self,
        frames_b64: list,
        return_components: bool = False
    ) -> Tuple[np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate fused embedding from 16 base64-encoded frames.

        Args:
            frames_b64: List of 16 base64-encoded frame strings
            return_components: If True, also return individual embeddings

        Returns:
            Tuple of (fused_embedding, description, video_embedding, text_embedding)
            If return_components is False, video_embedding and text_embedding are None
        """
        # Step 1: Generate video embedding
        print("  Generating video embedding...")
        video_embedding = self.encode_video(frames_b64)

        # Step 2: Generate description
        print("  Generating description...")
        description = self.generate_description(frames_b64)
        print(f"  Description: {description}")

        # Step 3: Generate text embedding
        print("  Generating text embedding...")
        text_embedding = self.encode_text(description)

        # Step 4: Fuse embeddings (average)
        fused_embedding = (video_embedding + text_embedding) / 2.0

        if return_components:
            return fused_embedding, description, video_embedding, text_embedding
        else:
            return fused_embedding, description, None, None

    def get_memory_usage(self) -> Optional[dict]:
        """Get GPU memory usage if available."""
        if self.device.type == "cuda":
            return {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            }
        return None

    @property
    def models_loaded(self) -> dict:
        """Return dictionary of which models are loaded."""
        return {
            "mineclip": self._mineclip_model is not None,
            "qwen": self._qwen_model is not None,
        }
