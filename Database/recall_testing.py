import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Database.client import ChromaClient
from mineclip.mineclip import MineCLIP
import numpy as np
import torch
import random
from pathlib import Path
import cv2
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import re
import csv

class RecallTesting():
    def __init__(self, video_dir=".data/chunked_dataset_with_embeddings"):
        self.db = self._init_database()
        self.video_dir = video_dir
        self.step_re = re.compile(r"^Step\s+\d+:\s*")
    
    # Initilizing the database
    def _init_database(self) -> ChromaClient:
        db = ChromaClient() # Database client

        if not db.connect():
            raise ConnectionError("Failed to connect to Chroma")
        
        return db
    
    def _get_random_chunk_path(self):
        base_dir = Path(self.video_dir)

        full_demo_folders = [p for p in base_dir.iterdir() if p.is_dir()]

        if not full_demo_folders:
            raise RuntimeError("No folders found")

        chosen_folder = random.choice(full_demo_folders)

        all_chunks = [p for p in chosen_folder.iterdir() if p.is_dir()]

        if not all_chunks:
            raise RuntimeError("No chunks found")

        chosen_chunk = random.choice(all_chunks)
        
        return chosen_chunk
    
    def save_recall_csv(self, recalls, sigmas, out_path="recall_vs_sigma.csv"):
        out_path = Path(out_path)

        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sigma", "recall"])
            for s, r in zip(sigmas, recalls):
                writer.writerow([s, r])
    
    # Plotting the recall
    def plot_recall(self, recalls, sigmas):
        plt.figure()
        plt.plot(sigmas, recalls, marker='o')
        plt.xlabel("Noise Sigma")
        plt.ylabel("Recall@1")
        plt.title("Retrieval Recall vs Embedding Noise")
        plt.xscale("log")
        plt.grid(True)
        plt.show()

    def strip_step_prefix(self, action_line: str) -> str:
        return self.step_re.sub("", action_line).strip()

    def _extract_action(self, action_sequence):
        action_list = action_sequence.strip().split("\n")
        last_action = action_list[-1]

        return self.strip_step_prefix(last_action)

    # Get the action to take
    def _extract_action_from_path(self, chunk_path):
        txt_path = chunk_path / "actions.txt"

        with open(txt_path, "r", encoding="utf-8") as f:
            action_sequence = f.read()

        return self._extract_action(action_sequence)

class EmbeddingNoise(RecallTesting):
    def __init__(self):
        super().__init__()

    def _add_gaussian_noise_embedding(self, e: np.ndarray, sigma: float, l2_normalize: bool = False) -> np.ndarray:
        """ 
        e: (D,) or (N,D)
        sigma: stddev in embedding units
        """
        e = e.astype(np.float32)
        noise = np.random.normal(0.0, sigma, size=e.shape).astype(np.float32)
        e2 = e + noise
        if l2_normalize:
            denom = np.linalg.norm(e2, axis=-1, keepdims=True) + 1e-12
            e2 = e2 / denom
        return e2
    
    # Load the embedding from the folder
    def _extract_embedding(self, chunk_path):
        path = chunk_path / "embedding.npy"
        if not path.exists():
            raise FileNotFoundError(path)

        embedding = np.load(path)

        return embedding
    
    # Load the many embeddings
    def _get_random_embeddings_and_actions(self, num_embeddings, chunk_path):
        embeddings_actions = []
        descript = f"Collecting embeddings"

        for _ in trange(num_embeddings, desc=descript):
            embedding = self._extract_embedding(chunk_path)
            action = self._extract_action_from_path(chunk_path)

            embeddings_actions.append((embedding, action))

        return embeddings_actions
    
    # Calculate the recall of this system
    def calc_recall(self, num_tests=100, noise_sigma=10, l2_normalize=False):
        chunk_path = self._get_random_chunk_path()
        embeddings_actions = self._get_random_embeddings_and_actions(num_tests, chunk_path)
        correct_retrievals = 0
        descript = f"Calculating accuracy for sigma {noise_sigma}"

        for embedding, action in tqdm(embeddings_actions, desc=descript):
            noisy_embedding = self._add_gaussian_noise_embedding(embedding, noise_sigma, l2_normalize)
            similar_actions = self.db.find_similar_actions(noisy_embedding, n_results=1)

            # TODO maybe include reranking for this testing
            chosen_action_sequence = similar_actions[0]['document']
            retrieved_action = self._extract_action(chosen_action_sequence)
            
            # The answer is correct
            if retrieved_action == action:
                correct_retrievals += 1

        return correct_retrievals / num_tests

class FrameNoise(RecallTesting):
    def __init__(self, target_size: tuple[int, int] = (160, 256)):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = self._load_embedding_model()
        self.target_size = target_size

    def _add_gaussian_noise_frames(self, frames: np.ndarray, sigma: float, clip: bool = True) -> np.ndarray:
        """
        frames: uint8 or float array shaped (16, H, W, C)
        sigma: noise stddev in pixel units if uint8 (0-255 scale), or in float scale if frames are [0,1]
        """
        x = frames.astype(np.float32)
        noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
        x_noisy = x + noise

        if clip:
            if frames.dtype == np.uint8:
                x_noisy = np.clip(x_noisy, 0, 255)
                return x_noisy.astype(np.uint8)
            else:
                # assume [0,1] floats
                x_noisy = np.clip(x_noisy, 0.0, 1.0)
                return x_noisy.astype(frames.dtype)
        return x_noisy.astype(frames.dtype)
    
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
    
    # Embed the 16 frames from the video
    def _embed_frames(self, frames) -> np.ndarray:
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
    
    def _extract_frames(self, chunk_path):
        pass

def test_embedding_noise():
    embedding_test = EmbeddingNoise()
    test_sigmas = [0.001, 0.01, 0.1, 1, 10, 100]
    recall_results = []
    
    for sigma in test_sigmas:
        recall_results.append(embedding_test.calc_recall(num_tests=1000, noise_sigma=sigma, l2_normalize=False))

    embedding_test.save_recall_csv(recall_results, test_sigmas)
    embedding_test.plot_recall(recall_results, test_sigmas)

if __name__ == "__main__":
    test_embedding_noise()