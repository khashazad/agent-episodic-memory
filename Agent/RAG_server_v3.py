"""
RAG Server v3: Fused Multimodal Embeddings

This RAG implementation uses fused multimodal embeddings that combine:
1. MineCLIP video embeddings from 16-frame observations
2. Qwen VLM-generated text descriptions
3. MineCLIP text embeddings from descriptions

The fused embedding is the average of video and text embeddings,
providing richer semantic representation for similarity matching.

Usage:
    from RAG_server_v3 import RAGFusedEmbedding

    rag = RAGFusedEmbedding()
    memory_str = rag.get_action(list_of_16_base64_frames)
"""

import sys
import os
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Database.client import ChromaClient
import torch
import numpy as np
import cv2
import base64
from PIL import Image


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


class RAGFusedEmbedding:
    """
    RAG v4: Uses fused multimodal embeddings for similarity search.

    This strategy:
    1. Generates a fused embedding from 16 frames (video + VLM description + text)
    2. Queries ChromaDB with the fused embedding
    3. Returns the next_action from the most similar past experience
    """

    def __init__(
        self,
        collection_name: str = "episodic_memory_v3",
        target_size: tuple = (160, 256)
    ):
        """Initialize the RAG with fused embedding support.

        Args:
            collection_name: Name of the ChromaDB collection
            target_size: Target frame resolution for MineCLIP
        """
        self.collection_name = collection_name
        self.target_size = target_size

        # Initialize database connection
        self.db = self._init_database()

        # Initialize fused embedding generator
        self.embedding_generator = FusedEmbeddingGenerator()

        # Action template for parsing
        self.action_template = {
            "attack": 0,
            "back": 0,
            "forward": 0,
            "jump": 0,
            "left": 0,
            "right": 0,
            "camera": [0.0, 0.0],
        }

    def _init_database(self) -> ChromaClient:
        """Initialize the database connection."""
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))

        db = ChromaClient(host=host, port=port, collection_name=self.collection_name)

        if not db.connect():
            raise ConnectionError(f"Failed to connect to Chroma collection: {self.collection_name}")

        return db

    def _parse_action(self, action_str: str) -> dict:
        """Parse action string from metadata into action dict.

        Args:
            action_str: JSON string of action dict

        Returns:
            Parsed action dictionary
        """
        try:
            action = json.loads(action_str)
            return action
        except json.JSONDecodeError:
            # Try to extract action-like patterns
            pass

        # Fallback: return template
        return self.action_template.copy()

    def _summarize_action(self, action: dict) -> str:
        """Build a readable description from the action dict."""
        parts = []

        for key in ["forward", "back", "left", "right", "jump", "attack"]:
            if action.get(key, 0):
                parts.append(key)

        cam = action.get("camera", [0.0, 0.0])
        if cam != [0.0, 0.0]:
            parts.append(f"camera({cam[0]}, {cam[1]})")

        if not parts:
            parts.append("idle")

        return " + ".join(parts)

    def get_action(self, obs_list_b64: list) -> str:
        """Query the database and return formatted memory context.

        Args:
            obs_list_b64: List of 16 base64-encoded frame strings

        Returns:
            Formatted memory context string for the LLM
        """
        # Generate fused embedding from frames
        fused_embedding, description = self.embedding_generator.generate_fused_embedding(obs_list_b64)

        # Query ChromaDB with fused embedding
        similar_results = self.db.find_similar_actions(
            fused_embedding,
            n_results=1,
            include_metadata=True
        )

        if not similar_results:
            return "No relevant episodic memory found."

        return self._format_memory_context(similar_results, description)

    def _format_memory_context(self, results: list, current_description: str) -> str:
        """Format memory context from retrieval results.

        Args:
            results: List of retrieval results from ChromaDB
            current_description: Description of current observation

        Returns:
            Formatted memory context string
        """
        result = results[0]
        metadata = result.get("metadata", {})
        document = result.get("document", "")
        similarity = result.get("similarity", 0.0)

        # Get next_action from metadata
        next_action_str = metadata.get("next_action", "{}")
        next_action = self._parse_action(next_action_str)
        action_summary = self._summarize_action(next_action)

        print(f"\t[RAG v4] Similarity: {similarity:.4f}")
        print(f"\t[RAG v4] Retrieved description: {document[:100]}...")
        print(f"\t[RAG v4] Suggested action: {action_summary}")

        return (
            "Similar past situation from episodic memory (fused multimodal match):\n"
            f"- Current observation: {current_description}\n"
            f"- Similar past observation: {document[:200]}\n"
            f"- Action taken in that situation: {action_summary}\n"
            f"- Match confidence: {similarity:.2%}\n"
            "Consider using a similar strategy if the situations align.\n"
        )


# =============================================================================
# Factory function for compatibility with existing code
# =============================================================================

def create_rag_v4() -> RAGFusedEmbedding:
    """Create a RAG instance with fused multimodal embeddings.

    Returns:
        RAGFusedEmbedding instance
    """
    print("Creating RAG v4 (fused multimodal) instance")
    print("  Collection: episodic_memory_v3")
    print("  Embedding: Fused (video + text average)")
    return RAGFusedEmbedding()


# For backward compatibility
RAG = RAGFusedEmbedding
