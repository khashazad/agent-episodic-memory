"""
FastAPI server for Qwen VL and MineCLIP models.

Provides REST API endpoints for:
- Video embedding generation (MineCLIP)
- Text embedding generation (MineCLIP)
- Description generation (Qwen VL)
- Fused embedding generation (video + text average)
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ModelServer.models import (
    VideoEmbeddingRequest,
    TextEmbeddingRequest,
    DescriptionRequest,
    FusedEmbeddingRequest,
    EmbeddingResponse,
    DescriptionResponse,
    FusedEmbeddingResponse,
    HealthResponse,
)
from ModelServer.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for model server."""
    manager = ModelManager()
    print(f"Model server starting on device: {manager.device}")
    print(f"Models will be loaded lazily on first request")
    yield
    print("Model server shutting down")


app = FastAPI(
    title="VLM Model Server",
    description="FastAPI server for Qwen2.5-VL and MineCLIP models",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - allow all origins (internal network, no auth)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager instance
manager = ModelManager()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns server status, device info, and which models are loaded.
    """
    return HealthResponse(
        status="healthy",
        device=str(manager.device),
        models_loaded=manager.models_loaded,
        memory_usage=manager.get_memory_usage(),
    )


@app.post("/video-embedding", response_model=EmbeddingResponse)
async def generate_video_embedding(request: VideoEmbeddingRequest):
    """
    Generate 512-dimensional video embedding from 16 frames using MineCLIP.

    Args:
        request: VideoEmbeddingRequest with 16 base64-encoded frames

    Returns:
        EmbeddingResponse with 512-dimensional embedding
    """
    try:
        embedding = manager.encode_video(request.frames_b64)
        return EmbeddingResponse(embedding=embedding.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video embedding failed: {str(e)}")


@app.post("/text-embedding", response_model=EmbeddingResponse)
async def generate_text_embedding(request: TextEmbeddingRequest):
    """
    Generate 512-dimensional text embedding using MineCLIP.

    Args:
        request: TextEmbeddingRequest with text to encode

    Returns:
        EmbeddingResponse with 512-dimensional embedding
    """
    try:
        embedding = manager.encode_text(request.text)
        return EmbeddingResponse(embedding=embedding.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text embedding failed: {str(e)}")


@app.post("/description", response_model=DescriptionResponse)
async def generate_description(request: DescriptionRequest):
    """
    Generate description from video frames using Qwen VLM.

    Args:
        request: DescriptionRequest with 16 base64-encoded frames

    Returns:
        DescriptionResponse with generated description text
    """
    try:
        description = manager.generate_description(request.frames_b64)
        return DescriptionResponse(description=description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Description generation failed: {str(e)}")


@app.post("/fused-embedding", response_model=FusedEmbeddingResponse)
async def generate_fused_embedding(request: FusedEmbeddingRequest):
    """
    Generate fused embedding (video + text average) from 16 frames.

    This is the primary endpoint that:
    1. Encodes video frames with MineCLIP
    2. Generates description with Qwen VLM
    3. Encodes description text with MineCLIP
    4. Returns average of video and text embeddings

    Args:
        request: FusedEmbeddingRequest with 16 base64-encoded frames

    Returns:
        FusedEmbeddingResponse with fused embedding, description, and component embeddings
    """
    try:
        fused_emb, description, video_emb, text_emb = manager.generate_fused_embedding(
            request.frames_b64,
            return_components=True
        )
        return FusedEmbeddingResponse(
            embedding=fused_emb.tolist(),
            description=description,
            video_embedding=video_emb.tolist() if video_emb is not None else None,
            text_embedding=text_emb.tolist() if text_emb is not None else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fused embedding failed: {str(e)}")


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))

    print(f"Starting VLM Model Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
