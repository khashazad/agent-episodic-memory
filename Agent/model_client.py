"""
Remote Model Client - HTTP client for the FastAPI model server.

This is a drop-in replacement for FusedEmbeddingGenerator that calls
the remote model server instead of loading models locally.
"""

import os
import time
from typing import Tuple, Optional

import numpy as np
import requests


class RemoteFusedEmbeddingGenerator:
    """
    Client for remote model server.

    Drop-in replacement for FusedEmbeddingGenerator that calls the remote
    FastAPI server for embeddings and descriptions.
    """

    def __init__(
        self,
        server_url: str = None,
        timeout: int = None,
        verify_connection: bool = True
    ):
        """
        Initialize the remote embedding generator client.

        Args:
            server_url: URL of the model server (default from MODEL_SERVER_URL env var)
            timeout: Request timeout in seconds (default from MODEL_SERVER_TIMEOUT env var)
            verify_connection: Whether to verify server connection on init
        """
        self.server_url = (
            server_url or
            os.environ.get("MODEL_SERVER_URL", "http://localhost:8080")
        ).rstrip("/")

        self.timeout = timeout or int(os.environ.get("MODEL_SERVER_TIMEOUT", "300"))

        print(f"RemoteFusedEmbeddingGenerator initialized")
        print(f"  Server URL: {self.server_url}")
        print(f"  Timeout: {self.timeout}s")

        if verify_connection:
            self._check_connection()

    def _check_connection(self) -> bool:
        """Verify server is reachable and return health status."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            if resp.status_code == 200:
                health = resp.json()
                print(f"  Connected to model server: {health['device']}")
                print(f"  Models loaded: {health['models_loaded']}")
                return True
            else:
                print(f"  Warning: Server returned status {resp.status_code}")
                return False
        except requests.exceptions.ConnectionError as e:
            print(f"  Warning: Could not connect to model server: {e}")
            return False
        except requests.exceptions.Timeout:
            print(f"  Warning: Connection to model server timed out")
            return False
        except Exception as e:
            print(f"  Warning: Unexpected error checking server: {e}")
            return False

    def generate_fused_embedding(self, frames_b64: list, max_retries: int = 3) -> Tuple[np.ndarray, str]:
        """
        Generate fused embedding from 16 base64-encoded frames.

        This is the main method matching FusedEmbeddingGenerator's interface.

        Args:
            frames_b64: List of 16 base64-encoded frame strings
            max_retries: Number of retries on timeout (first request may take long for model loading)

        Returns:
            Tuple of (fused_embedding, description):
                - fused_embedding: 512-dimensional numpy array
                - description: Generated text description
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                # Use longer timeout for first attempt (model loading)
                current_timeout = self.timeout * 2 if attempt == 0 else self.timeout

                resp = requests.post(
                    f"{self.server_url}/fused-embedding",
                    json={"frames_b64": frames_b64},
                    timeout=current_timeout
                )

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Model server error: {resp.status_code} - {resp.text}"
                    )

                data = resp.json()
                embedding = np.array(data["embedding"], dtype=np.float32)
                description = data["description"]

                return embedding, description

            except requests.exceptions.Timeout as e:
                last_error = e
                print(f"  Request timed out (attempt {attempt + 1}/{max_retries}). Retrying...")
                time.sleep(2)
            except requests.exceptions.ConnectionError as e:
                last_error = e
                print(f"  Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(5)

        raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")

    def encode_video(self, frames_b64: list) -> np.ndarray:
        """
        Generate video embedding only using MineCLIP.

        Args:
            frames_b64: List of 16 base64-encoded frame strings

        Returns:
            512-dimensional numpy array
        """
        resp = requests.post(
            f"{self.server_url}/video-embedding",
            json={"frames_b64": frames_b64},
            timeout=self.timeout
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Model server error: {resp.status_code} - {resp.text}"
            )

        return np.array(resp.json()["embedding"], dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate text embedding using MineCLIP.

        Args:
            text: Description text

        Returns:
            512-dimensional numpy array
        """
        resp = requests.post(
            f"{self.server_url}/text-embedding",
            json={"text": text},
            timeout=self.timeout
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Model server error: {resp.status_code} - {resp.text}"
            )

        return np.array(resp.json()["embedding"], dtype=np.float32)

    def generate_description(self, frames_b64: list) -> str:
        """
        Generate description only using Qwen VLM.

        Args:
            frames_b64: List of 16 base64-encoded frame strings

        Returns:
            Generated description string
        """
        resp = requests.post(
            f"{self.server_url}/description",
            json={"frames_b64": frames_b64},
            timeout=self.timeout
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Model server error: {resp.status_code} - {resp.text}"
            )

        return resp.json()["description"]

    def get_health(self) -> dict:
        """
        Get health status from the model server.

        Returns:
            Dictionary with status, device, models_loaded, memory_usage
        """
        resp = requests.get(f"{self.server_url}/health", timeout=10)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Model server error: {resp.status_code} - {resp.text}"
            )

        return resp.json()
