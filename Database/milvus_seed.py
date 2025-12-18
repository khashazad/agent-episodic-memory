#!/usr/bin/env python3
"""
Seed 3 Milvus collections with the same data but different index methods:
  - HNSW
  - IVF_PQ
  - DISKANN (DiskANN)

Install:
  pip install pymilvus numpy tqdm

Run:
  python3 seed_milvus_3indexes.py --dataset-dir .data/chunked_dataset_with_embeddings --reset-collections
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_chunk_dirs(dataset_dir: Path) -> List[Path]:
    chunk_dirs: List[Path] = []
    if not dataset_dir.exists():
        return chunk_dirs

    for episode_dir in dataset_dir.iterdir():
        if episode_dir.is_dir():
            for chunk_dir in episode_dir.iterdir():
                if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk_"):
                    chunk_dirs.append(chunk_dir)

    return sorted(chunk_dirs)


def infer_dim(chunk_dirs: List[Path]) -> int:
    for d in chunk_dirs:
        p = d / "embedding.npy"
        if p.exists():
            emb = np.load(p).reshape(-1)
            return int(emb.shape[0])
    raise RuntimeError("Could not infer embedding dimension: no embedding.npy found.")


def load_chunk_data(chunk_dir: Path) -> Optional[Dict]:
    try:
        embedding_path = chunk_dir / "embedding.npy"
        actions_path = chunk_dir / "actions.txt"

        if not embedding_path.exists():
            return None
        if not actions_path.exists():
            return None

        embedding = np.load(embedding_path).reshape(-1).astype(np.float32)

        with open(actions_path, "r") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

        action_descriptions: List[str] = []
        for line in lines:
            if line.startswith("Step"):
                desc = line.split(": ", 1)[1] if ": " in line else line
                action_descriptions.append(desc)
            else:
                action_descriptions.append(line)

        return {"embedding": embedding, "document": "\n".join(action_descriptions)}

    except Exception as e:
        logger.error(f"Failed loading {chunk_dir}: {e}")
        return None


def connect_milvus(host: str, port: int, alias: str = "default") -> None:
    connections.connect(alias=alias, host=host, port=str(port))
    _ = utility.list_collections(using=alias)
    logger.info(f"Connected to Milvus at {host}:{port}")


def make_collection(
    alias: str,
    name: str,
    dim: int,
    reset: bool,
) -> Collection:
    if utility.has_collection(name, using=alias):
        if reset:
            utility.drop_collection(name, using=alias)
            logger.info(f"Dropped existing collection: {name}")
        else:
            c = Collection(name, using=alias)
            logger.info(f"Using existing collection: {name}")
            return c

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=512,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
        ),
        FieldSchema(
            name="document",
            dtype=DataType.VARCHAR,
            max_length=65535,
        ),
    ]
    schema = CollectionSchema(fields=fields, description="Minecraft episodic memory")
    c = Collection(name=name, schema=schema, using=alias)
    logger.info(f"Created collection: {name} (dim={dim})")
    return c


def create_index_for_method(
    collection: Collection,
    method: str,
    metric_type: str,
    # HNSW
    hnsw_m: int,
    hnsw_ef_construction: int,
    # IVF_PQ
    ivf_nlist: int,
    ivf_pq_m: int,
    ivf_pq_nbits: int,
) -> None:
    method_u = method.upper()

    if method_u == "HNSW":
        index_params = {
            "index_type": "HNSW",
            "metric_type": metric_type,
            "params": {"M": hnsw_m, "efConstruction": hnsw_ef_construction},
        }
        collection.create_index("embedding", index_params)

    elif method_u in {"IVF-PQ", "IVF_PQ"}:
        index_params = {
            "index_type": "IVF_PQ",
            "metric_type": metric_type,
            "params": {"nlist": ivf_nlist, "m": ivf_pq_m, "nbits": ivf_pq_nbits},
        }
        collection.create_index("embedding", index_params)

    elif method_u in {"DISKANN", "DISK_ANN"}:
        # Milvus docs: DISKANN index doesn't require index params in create_index().
        # (Some build knobs are configured in milvus.yaml.)
        index_params = {
            "index_type": "DISKANN",
            "metric_type": metric_type,
            "params": {},
        }
        collection.create_index("embedding", index_params)

    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Created index on {collection.name}: {index_params}")


def seed_collection(
    collection: Collection,
    chunk_dirs: List[Path],
    batch_size: int,
) -> int:
    ids: List[str] = []
    embs: List[List[float]] = []
    docs: List[str] = []
    inserted = 0

    for chunk_dir in tqdm(chunk_dirs, desc=f"Seeding {collection.name}"):
        cd = load_chunk_data(chunk_dir)
        if cd is None:
            continue

        episode_name = chunk_dir.parent.name
        chunk_name = chunk_dir.name
        chunk_id = f"{episode_name}_{chunk_name}"

        ids.append(chunk_id)
        embs.append(cd["embedding"].tolist())
        docs.append(cd["document"])

        if len(ids) >= batch_size:
            collection.insert([ids, embs, docs])
            inserted += len(ids)
            ids, embs, docs = [], [], []

    if ids:
        collection.insert([ids, embs, docs])
        inserted += len(ids)

    collection.flush()
    collection.load()
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Seed Milvus into 3 collections with HNSW, IVF_PQ, DISKANN")
    parser.add_argument("--dataset-dir", type=str, default=".data/chunked_dataset_with_embeddings")
    parser.add_argument("--milvus-host", type=str, default="localhost")
    parser.add_argument("--milvus-port", type=int, default=19530)
    parser.add_argument("--alias", type=str, default="default")

    parser.add_argument("--base-collection-name", type=str, default="episodic_memory")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--reset-collections", action="store_true")

    parser.add_argument("--metric-type", type=str, default="L2", choices=["L2", "IP", "COSINE"])

    # HNSW build params
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)

    # IVF_PQ build params
    parser.add_argument("--ivf-nlist", type=int, default=2048)
    parser.add_argument("--ivf-pq-m", type=int, default=32)
    parser.add_argument("--ivf-pq-nbits", type=int, default=8)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    chunk_dirs = get_chunk_dirs(dataset_dir)
    if not chunk_dirs:
        logger.error(f"No chunk_* dirs found under {dataset_dir}")
        sys.exit(1)

    dim = infer_dim(chunk_dirs)
    logger.info(f"Inferred embedding dim = {dim}")

    connect_milvus(args.milvus_host, args.milvus_port, alias=args.alias)

    methods = ["HNSW", "IVF_PQ", "DISKANN"]
    for method in methods:
        suffix = method.lower().replace("-", "_")
        name = f"{args.base_collection_name}_{suffix}"

        c = make_collection(args.alias, name, dim, reset=args.reset_collections)

        # Build index first (recommended before heavy insert? either works; this is simple/reliable for seeding)
        # If you prefer: insert first then create_index — also valid. Keeping consistent across 3 collections.
        create_index_for_method(
            c,
            method=method,
            metric_type=args.metric_type,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            ivf_nlist=args.ivf_nlist,
            ivf_pq_m=args.ivf_pq_m,
            ivf_pq_nbits=args.ivf_pq_nbits,
        )

        inserted = seed_collection(c, chunk_dirs, batch_size=args.batch_size)
        logger.info(f"{name}: inserted {inserted} vectors")

    logger.info("Done.")


if __name__ == "__main__":
    main()