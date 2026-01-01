# Agent Episodic Memory

A Minecraft-playing agent with episodic memory using Retrieval-Augmented Generation (RAG). The agent is tasked to collect wood in the MineRL environment using vision-language models and vector database retrieval for episodic memory.

## Overview

This project implements an Minecraft agent that learns from past experiences using episodic memory. The agent uses:

- **MineCLIP**: A CLIP-based vision model fine-tuned for Minecraft video understanding
- **Qwen2.5-VL**: A vision-language model for generating descriptions and action sequences
- **ChromaDB**: Vector database for storing and retrieving episodic memories
- **LangChain/LangGraph**: LLM orchestration framework for agent decision-making

The agent can operate in multiple modes:
- **Baseline**: No memory, pure LLM-based decision making
- **RAG v1**: Text-based retrieval using action or description embeddings
- **RAG v4 (Multimodal)**: Fused video + text embeddings for improved retrieval

## Project Structure

```
|── .data                     # Data folders (MineRL dataset, database seed, ...)
|── .ckpts                    # Model weights
├── Agent/                    # Agent implementations
│   ├── agent.py              # OpenAI GPT-based agent
│   ├── agent_local.py        # Local HuggingFace model agent
│   ├── agent_multimodal.py   # Multimodal fused embedding agent
│   ├── RAG_server.py         # RAG retrieval server
│   └── utils/                # Agent utilities and prompts
├── MineCLIP/                 # MineCLIP vision model
│   ├── mineclip/             # Model architecture
│   └── utils/                # Vision utilities
├── DatasetProcessing/        # Data pipeline
│   ├── pipeline/             # 6-step processing pipeline
│   ├── run_pipeline.py       # Main pipeline runner
│   └── process_sliding_window_pipeline.py
├── Database/                 # Vector database management
│   ├── seed_collections.py   # ChromaDB seeding
│   └── recall_testing.py     # Retrieval evaluation
├── EnvServer/                # MineRL environment servers
│   ├── env_server.py         # Single environment
│   └── multi_env_server.py   # Multi-environment support
├── Scripts/                  # SLURM batch scripts
└── docker-compose.yml        # Container orchestration
```

## Installation

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- CUDA-capable GPU (recommended: 24GB+ VRAM for multimodal agent)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agent-episodic-memory.git
cd agent-episodic-memory
```

2. Install dependencies:
```bash
pip install -e .
# or using uv
uv sync
```

3. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Download MineCLIP model weights (see [Data & Model Weights](#data--model-weights))

5. Start the Docker services:
```bash
docker-compose up -d
```

## Data & Model Weights

### MineCLIP Checkpoints

Download the MineCLIP model checkpoints and place them in the `.ckpts/` directory:

| Model | Description | Download |
|-------|-------------|----------|
| `attn.pth` | Attention-based temporal pooling (634MB) | [MineCLIP GitHub](https://github.com/MineDojo/MineCLIP) |
| `avg.pth` | Average temporal pooling (601MB) | [MineCLIP GitHub](https://github.com/MineDojo/MineCLIP) |

The checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1_dbnAN6bo7sbfgamejRAW8YptUG4ak2Q?usp=sharing).


### MineRL Dataset

The project uses the MineRLTreechop-v0 dataset which can be downloaded [here](https://drive.google.com/drive/folders/1_dbnAN6bo7sbfgamejRAW8YptUG4ak2Q?usp=sharing).

| Dataset | Description | Download |
|---------|-------------|----------|
| MineRLTreechop-v0 | Human demonstrations of tree chopping | [MineRL Dataset](https://minerl.io/dataset/) |

**Dataset structure after download:**
```
.data/
└── MineRLTreechop-v0/
    ├── v3_absolute_grape_changeling-1_1050-2666/
    │   ├── rendered.npz     # Video frames
    │   └── metadata.json    # Actions and rewards
    └── ... (more episodes)
```

## Running the Agent

### Configuration

Key environment variables in `.env`:

```bash
# LLM Configuration
LOCAL_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
USE_OPENAI_LLM=false  # Set to true for OpenAI models

# Agent Settings
USE_RAG=true          # Enable episodic memory
MAX_FRAMES=500        # Maximum steps per episode
TEST_RUNS=5           # Number of evaluation runs

# Server Configuration
MINERL_SERVER_URL=http://127.0.0.1:5002
CHROMA_HOST=http://127.0.0.1:8000
```

### Running Agents

**Local HuggingFace Agent:**
```bash
python Agent/agent_local.py
```

**OpenAI Agent:**
```bash
export OPENAI_API_KEY=sk-your-key-here
python Agent/agent.py
```

**Multimodal Agent (with fused embeddings):**
```bash
python Agent/agent_multimodal.py
```

### SLURM Cluster Scripts

For HPC environments:

```bash
# Run local agent
sbatch Scripts/run_local_agent.sbatch

# Run multimodal agent (requires A6000 or better)
sbatch Scripts/run_multimodal_agent.sbatch

# Run RAG experiments as array job
sbatch Scripts/run_rag_array.sbatch
```

## Data Processing Pipeline

The 6-step pipeline processes raw MineRL episodes into vector database entries:

1. **Step 1**: Sliding window chunking (16-frame windows, stride 8)
2. **Step 2**: Video embedding using MineCLIP
3. **Step 3**: Description generation using Qwen VLM
4. **Step 4**: Text embedding using MineCLIP
5. **Step 5**: Action sequence generation using Qwen VLM
6. **Step 6**: CSV export for ChromaDB

### Running the Pipeline

```bash
# Full pipeline
python DatasetProcessing/run_pipeline.py \
    --input-dir .data/MineRLTreechop-v0 \
    --output-dir .data/pipeline_output

# With specific episodes
python DatasetProcessing/run_pipeline.py \
    --input-dir .data/MineRLTreechop-v0 \
    --max-episodes 10

# Sliding window pipeline (recommended)
python DatasetProcessing/process_sliding_window_pipeline.py \
    --input-dir .data/MineRLTreechop-v0 \
    --output-dir .data/sliding_window_dataset
```

### Individual Pipeline Steps

```bash
# Step 2: Embed videos
python DatasetProcessing/pipeline/step2_video_embedding.py \
    --input-dir .data/chunked_dataset \
    --checkpoint .ckpts/attn.pth

# Step 3: Generate descriptions
python DatasetProcessing/pipeline/step3_llm_description.py \
    --input-dir .data/chunked_dataset_with_embeddings

# Export to CSV
python DatasetProcessing/export_to_csv.py
```

## Docker Services

```bash
# Start all services
docker-compose up -d

# Start only ChromaDB
docker-compose up -d chromadb

# Start MineRL multi-environment server
docker-compose up -d minerl-multi-env

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| ChromaDB | 8000 | Vector database HTTP API |
| MineRL Multi-Env | 5002 | Multi-environment REST API |

## RAG Configurations

The project supports 4 RAG versions (configured in `rag_configs.json`):

| Version | Embedding Method | Description |
|---------|-----------------|-------------|
| v1 | Action-based | Uses last action from metadata |
| v2 | Full description | LLM-generated episode description |
| v3 | Chunk description | LLM-generated chunk-level description |
| v4 | Fused multimodal | Combined video + text embeddings (best performance) |

## Evaluation

Results are saved to `Agent/Results/` with metrics including:
- Wood collected per episode
- Success rate
- Average episode length
- Retrieval statistics (when using RAG)
