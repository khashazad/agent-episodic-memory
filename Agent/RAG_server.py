import sys
import os
import json
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Database.client import ChromaClient
import torch
from MineCLIP.mineclip import MineCLIP
import numpy as np
import cv2
import base64
import re


# =============================================================================
# Configuration Loading
# =============================================================================

def load_rag_config(config_name: str) -> dict:
    """Load RAG configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), "rag_configs.json")

    with open(config_path, "r") as f:
        configs = json.load(f)

    if config_name not in configs:
        raise ValueError(f"Unknown RAG config: {config_name}. Available: {list(configs.keys())}")

    return configs[config_name]


# =============================================================================
# Base RAG Class
# =============================================================================

class RAGBase(ABC):
    """Abstract base class for RAG implementations."""

    def __init__(self, collection_name: str, target_size: tuple[int, int] = (160, 256)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.collection_name = collection_name
        self.db = self._init_database()
        self.embedding_model = self._load_embedding_model()
        self.target_size = target_size

    def _init_database(self) -> ChromaClient:
        """Initialize the database connection."""
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))

        db = ChromaClient(host=host, port=port, collection_name=self.collection_name)

        if not db.connect():
            raise ConnectionError(f"Failed to connect to Chroma collection: {self.collection_name}")

        return db

    def _load_embedding_model(self) -> MineCLIP:
        """Load the MineCLIP embedding model."""
        checkpoint_path = ".ckpts/attn.pth"

        model = MineCLIP(
            arch="vit_base_p16_fz.v2.t2",
            resolution=(160, 256),
            pool_type="attn.d2.nh8.glusw",
            image_feature_dim=512,
            mlp_adapter_spec="v0-2.t0",
            hidden_dim=512,
        ).to(self.device)

        model.load_ckpt(checkpoint_path, strict=False)
        model.eval()
        return model

    def _resize_frames(self, frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize frames to target resolution for MineCLIP."""
        resized = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (target_size[1], target_size[0]))
            resized.append(resized_frame)
        return np.array(resized)

    def _b64_to_numpy(self, b64_string: str) -> np.ndarray:
        """Convert a base64-encoded PNG/JPEG to a NumPy RGB image."""
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _load_frames_from_b64_list(self, b64_list: list[str]) -> np.ndarray:
        """Convert a list of base64-encoded frames into a stacked NumPy array."""
        frames = []
        for b64_string in b64_list:
            frame = self._b64_to_numpy(b64_string)
            frames.append(frame)
        return np.stack(frames, axis=0)

    def _embed_frames(self, obs_list_b64: list[str]) -> np.ndarray:
        """Embed the 16 frames from the video."""
        frames = self._load_frames_from_b64_list(obs_list_b64)
        frames_resized = self._resize_frames(frames, self.target_size)

        # [16, H, W, C] -> [1, 16, C, H, W]
        frames_tensor = (
            torch.from_numpy(frames_resized)
            .permute(0, 3, 1, 2)      # [16, 3, 160, 256]
            .unsqueeze(0)             # [1, 16, 3, 160, 256]
            .float()
            .to(self.device)
        )

        torch_embedding = self.embedding_model.encode_video(frames_tensor)
        embedding = torch_embedding.cpu().detach().numpy()

        return embedding[0]

    @abstractmethod
    def _format_memory_context(self, results: list[dict]) -> str:
        """Format the retrieved memory into context for the LLM."""
        pass

    def get_action(self, obs_list_b64: list[str]) -> str:
        """Query the database and return formatted memory context."""
        embedding = self._embed_frames(obs_list_b64)

        # Query with metadata for action-based strategies
        similar_results = self.db.find_similar_actions(
            embedding,
            n_results=1,
            include_metadata=True
        )

        if not similar_results:
            return "No relevant episodic memory found."

        return self._format_memory_context(similar_results)


# =============================================================================
# RAG Strategy: Action-Based (v1)
# =============================================================================

class RAGActionBased(RAGBase):
    """
    RAG v1: Uses the last action from action_descriptions metadata.
    Collection: episodic_memory_v1
    """

    def __init__(self, target_size: tuple[int, int] = (160, 256)):
        super().__init__(collection_name="episodic_memory_v1", target_size=target_size)
        self.action_template = {
            "attack": 0,
            "back": 0,
            "forward": 0,
            "jump": 0,
            "left": 0,
            "right": 0,
            "camera": [0.0, 0.0],
        }
        self.camera_re = re.compile(r"camera\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)")

    def parse_action_line(self, line: str) -> dict:
        """Parse a single action line into a dict."""
        line = line.strip()
        action = self.action_template.copy()
        action["camera"] = action["camera"].copy()

        cam = self.camera_re.search(line)
        if cam:
            pitch = float(cam.group(1))
            yaw = float(cam.group(2))
            action["camera"] = [pitch, yaw]
            line = self.camera_re.sub("", line)

        parts = [p.strip() for p in line.split("+") if p.strip()]

        for p in parts:
            if p in action:
                if p == "camera":
                    continue
                action[p] = 1

        return action

    def _get_last_action(self, action_descriptions: str) -> dict:
        """Extract the last action from action_descriptions."""
        action_list = action_descriptions.strip().split("\n")
        last_action = action_list[-1]
        return self.parse_action_line(last_action)

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

    def _format_memory_context(self, results: list[dict]) -> str:
        """Format memory using last action from metadata."""
        result = results[0]
        metadata = result.get("metadata", {})
        action_descriptions = metadata.get("action_descriptions", "")

        if not action_descriptions:
            return "No action information available in memory."

        action_dict = self._get_last_action(action_descriptions)
        action_str = self._summarize_action(action_dict)

        return (
            "Similar past situation from episodic memory:\n"
            f"- Last action taken in that sequence: {action_str}\n"
            "You may use a similar strategy here if it seems appropriate.\n"
        )


# =============================================================================
# RAG Strategy: Document Full Episode (v2)
# =============================================================================

class RAGDocumentFull(RAGBase):
    """
    RAG v2: Uses the full episode LLM description from document text.
    Collection: episodic_memory_v1
    """

    def __init__(self, target_size: tuple[int, int] = (160, 256)):
        super().__init__(collection_name="episodic_memory_v1", target_size=target_size)

    def _format_memory_context(self, results: list[dict]) -> str:
        """Format memory using document text (full episode description)."""
        result = results[0]
        document_text = result.get("document", "")

        if not document_text:
            return "No episode description available in memory."

        return (
            "Similar past situation from episodic memory:\n"
            f"Episode description: {document_text.strip()}\n"
            "Consider this context when deciding your next action.\n"
        )


# =============================================================================
# RAG Strategy: Document Chunk (v3)
# =============================================================================

class RAGDocumentChunk(RAGBase):
    """
    RAG v3: Uses the chunk LLM description from document text.
    Collection: episodic_memory_v2
    """

    def __init__(self, target_size: tuple[int, int] = (160, 256)):
        super().__init__(collection_name="episodic_memory_v2", target_size=target_size)

    def _format_memory_context(self, results: list[dict]) -> str:
        """Format memory using document text (chunk description)."""
        result = results[0]
        document_text = result.get("document", "")

        if not document_text:
            return "No chunk description available in memory."

        return (
            "Similar past situation from episodic memory:\n"
            f"Suggested action: {document_text.strip()}\n"
            "Consider this guidance when deciding your next action.\n"
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_rag(config_name: str = "v1") -> RAGBase:
    """
    Factory function to create the appropriate RAG instance based on config.

    Args:
        config_name: One of "v1", "v2", "v3"

    Returns:
        RAG instance of the appropriate type
    """
    config = load_rag_config(config_name)
    strategy_name = config["name"]

    strategy_map = {
        "action_based": RAGActionBased,
        "document_full": RAGDocumentFull,
        "document_chunk": RAGDocumentChunk,
    }

    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown RAG strategy: {strategy_name}")

    rag_class = strategy_map[strategy_name]
    print(f"Creating RAG instance: {rag_class.__name__} (config: {config_name})")
    print(f"  Collection: {config['collection']}")
    print(f"  Description: {config['description']}")

    return rag_class()


# =============================================================================
# Legacy RAG class for backward compatibility
# =============================================================================

class RAG(RAGActionBased):
    """
    Legacy RAG class for backward compatibility.
    Defaults to action-based strategy (v1).
    """
    pass
