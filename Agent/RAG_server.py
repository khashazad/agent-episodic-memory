import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Database.client import ChromaClient
import torch
from MineCLIP.mineclip import MineCLIP
import numpy as np
import cv2
import base64
import re

class RAG():
    def __init__(self, target_size: tuple[int, int] = (160, 256)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db = self._init_database()
        self.embedding_model = self._load_embedding_model()
        self.target_size = target_size
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

    # Initilizing the database
    def _init_database(self) -> ChromaClient:
        db = ChromaClient() # Database client

        if not db.connect():
            raise ConnectionError("Failed to connect to Chroma")

        return db

    # Loads the embedding model for use
    # TODO change this to work with many embeddings
    def _load_embedding_model(self) -> MineCLIP:
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
        # decode base64 → bytes
        img_bytes = base64.b64decode(b64_string)

        # convert bytes → uint8 array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        # decode PNG/JPEG buffer → BGR image
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # convert to RGB to match your PNG loader
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return img_rgb

    def _load_frames_from_b64_list(self, b64_list: list[str]) -> np.ndarray:
        """Convert a list of base64-encoded frames into a stacked NumPy array."""
        frames = []

        for b64_string in b64_list:
            frame = self._b64_to_numpy(b64_string)
            frames.append(frame)

        return np.stack(frames, axis=0)

    # Embed the 16 frames from the video
    def _embed_frames(self, obs_list_b64: list[str]) -> np.ndarray:
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

        # Encode batch
        torch_embedding = self.embedding_model.encode_video(frames_tensor)
        embedding = torch_embedding.cpu().detach().numpy()

        return embedding[0]

    # Gets the action sequence
    def parse_action_line(self, line: str) -> dict:
        line = line.strip()

        # Start with a fresh copy of the action template
        action = self.action_template.copy()
        action["camera"] = action["camera"].copy()

        # Check for camera(...) first
        cam = self.camera_re.search(line)
        if cam:
            pitch = float(cam.group(1))
            yaw = float(cam.group(2))
            action["camera"] = [pitch, yaw]
            # Remove from line so only discrete actions remain
            line = self.camera_re.sub("", line)

        # Extract discrete actions: split on '+', strip whitespace
        parts = [p.strip() for p in line.split("+") if p.strip()]

        for p in parts:
            if p in action:
                # Ignore camera because handled above
                if p == "camera":
                    continue
                action[p] = 1

        return action

    # Returns the action sequence
    def _get_returned_action(self, chosen_action_sequence):
        # TODO fix for reranking
        action_list = chosen_action_sequence.strip().split("\n")
        last_action = action_list[-1]

        # Dict representation
        action_dict = self.parse_action_line(last_action)

        return action_dict

    # Turns the memory chunk into context
    def _summarize_memory_chunk(self, action: dict) -> str:
        # Build a readable description from the action dict
        parts = []

        for key in ["forward", "back", "left", "right", "jump", "attack"]:
            if action.get(key, 0):
                parts.append(key)

        cam = action.get("camera", [0.0, 0.0])
        if cam != [0.0, 0.0]:
            parts.append(f"camera({cam[0]}, {cam[1]})")

        if not parts:
            parts.append("idle")

        act_str = " + ".join(parts)

        return (
            "Similar past situation from episodic memory:\n"
            f"- Last action taken in that sequence: {act_str}\n"
            "You may use a similar strategy here if it seems appropriate.\n"
        )

    # Takes in the raw b64 frames and returns an action
    # TODO add a reranking system here
    def get_action(self, obs_list_b64) -> str:
        # Get the embedding of the frame
        embedding = self._embed_frames(obs_list_b64)

        # TODO only returning the top result rn
        similar_actions = self.db.find_similar_actions(embedding, n_results=1)

        chosen_action_sequence = similar_actions[0]['document']

        chosen_action_dict = self._get_returned_action(chosen_action_sequence)

        memory_chunk = self._summarize_memory_chunk(chosen_action_dict)

        return memory_chunk
