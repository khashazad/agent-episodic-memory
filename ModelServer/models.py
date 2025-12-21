"""
Pydantic models for FastAPI request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# =============================================================================
# Request Models
# =============================================================================

class VideoEmbeddingRequest(BaseModel):
    """Request for video embedding generation using MineCLIP."""
    frames_b64: List[str] = Field(
        ...,
        description="List of 16 base64-encoded PNG/JPEG frames",
        min_length=16,
        max_length=16
    )


class TextEmbeddingRequest(BaseModel):
    """Request for text embedding generation using MineCLIP."""
    text: str = Field(
        ...,
        description="Text to encode",
        min_length=1
    )


class DescriptionRequest(BaseModel):
    """Request for VLM description generation using Qwen."""
    frames_b64: List[str] = Field(
        ...,
        description="List of 16 base64-encoded frames (8 will be sampled)",
        min_length=16,
        max_length=16
    )


class FusedEmbeddingRequest(BaseModel):
    """Request for fused embedding generation (video + text average)."""
    frames_b64: List[str] = Field(
        ...,
        description="List of 16 base64-encoded frames",
        min_length=16,
        max_length=16
    )


# =============================================================================
# Response Models
# =============================================================================

class EmbeddingResponse(BaseModel):
    """Response containing a 512-dimensional embedding."""
    embedding: List[float] = Field(
        ...,
        description="512-dimensional embedding vector"
    )


class DescriptionResponse(BaseModel):
    """Response containing VLM-generated description."""
    description: str = Field(
        ...,
        description="VLM-generated description of the video frames"
    )


class FusedEmbeddingResponse(BaseModel):
    """Response containing fused embedding and description."""
    embedding: List[float] = Field(
        ...,
        description="512-dimensional fused embedding (average of video + text)"
    )
    description: str = Field(
        ...,
        description="VLM-generated description"
    )
    video_embedding: Optional[List[float]] = Field(
        None,
        description="Raw video embedding (512-dim)"
    )
    text_embedding: Optional[List[float]] = Field(
        None,
        description="Raw text embedding (512-dim)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(
        ...,
        description="Server status (healthy/unhealthy)"
    )
    device: str = Field(
        ...,
        description="Compute device being used (cuda/mps/cpu)"
    )
    models_loaded: dict = Field(
        ...,
        description="Dictionary showing which models are loaded"
    )
    memory_usage: Optional[dict] = Field(
        None,
        description="GPU memory usage if available"
    )
