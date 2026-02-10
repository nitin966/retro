# RETRO - Retrieval-Enhanced Transformer

A PyTorch implementation of the **RETRO** (Retrieval-Enhanced Transformer) architecture from the paper [*Improving Language Models by Retrieving from Trillions of Tokens*](https://arxiv.org/abs/2112.05329) (Borgeaud et al., 2021).

RETRO augments a standard autoregressive language model with a retrieval mechanism that fetches semantically similar text chunks from an external corpus during both training and inference, enabling improved language modeling without scaling model parameters.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data Preparation Pipeline](#data-preparation-pipeline)
- [Training](#training)
- [Evaluation](#evaluation)
- [Text Generation](#text-generation)
- [Testing](#testing)
- [Configuration Reference](#configuration-reference)
- [Citation](#citation)

## Architecture Overview

The model consists of two main components:

**Encoder** - Processes retrieved neighbor chunks through self-attention and cross-attention layers:
- Configurable number of layers (default: 2)
- Cross-attention to decoder hidden states at specified layers
- RMSNorm normalization
- Relative position bias with linear and log-linear buckets

**Decoder** - The main autoregressive language model with retrieval augmentation:
- Standard self-attention layers
- **Chunked Cross-Attention (CCA)** layers that attend to encoded neighbor representations
- CCA integrates retrieval information at configurable layer positions
- Supports HuggingFace `GenerationMixin` for flexible text generation

The retrieval pipeline uses **Sentence Transformers** to embed text chunks and **FAISS** for efficient nearest-neighbor search over the embedding index.

## Project Structure

```
retro/
├── src/
│   ├── modeling_retro.py          # Core RETRO model (encoder, decoder, attention)
│   ├── dataset_retro.py           # Dataset classes for chunked sequences and neighbors
│   ├── train_retro.py             # Training script (PyTorch Lightning)
│   ├── evaluate_retro.py          # Evaluation script
│   ├── generate_retro.py          # Interactive text generation
│   ├── retrieval.py               # Retriever interfaces (FAISS, HTTP, caching)
│   ├── index_service.py           # FastAPI service for vector search
│   ├── faiss_http_server.py       # HTTP wrapper for FAISS indices
│   ├── test_modeling_retro.py     # Unit tests
│   └── data/
│       ├── tokenize_and_chunk.py  # Tokenize documents into fixed-size chunks
│       ├── embed_chunks.py        # Generate Sentence Transformer embeddings
│       ├── train_faiss_index.py   # Train FAISS index structure
│       ├── build_faiss_index.py   # Populate FAISS index with embeddings
│       └── retrieve_neighbours.py # Retrieve nearest-neighbor chunks
├── data/
│   ├── datasets/                  # Dataset storage and preparation guides
│   └── model/                     # Pre-trained model storage
├── Dockerfile                     # Container image definition
├── start.sh                       # Docker launch script
└── bash.bashrc                    # Container shell configuration
```

## Requirements

- Python 3.9
- PyTorch 2.4.1
- PyTorch Lightning 1.7.4
- Transformers 4.21.0
- Sentence Transformers 2.2.2
- FAISS 1.7.4 (CPU or GPU)
- einops 0.6.0
- pytest 7.2.1
- matplotlib 3.6.3
- seaborn 0.12.2

## Setup

### Using Docker (recommended)

The repository includes a Docker-based environment that installs all dependencies automatically.

```bash
# Basic setup (CPU only)
./start.sh

# With GPU support
./start.sh --gpu

# With an external data directory mounted
./start.sh --gpu -v /path/to/data:/data

# Run in detached mode
./start.sh --gpu -d
```

You can also configure the environment via a `.env` file in the project root:

```bash
DOCKER_IMAGE_NAME=tobias/retro-analysis   # Docker image name
DOCKER_WORKSPACE_PATH=/workspace           # Container workspace path
DATA_DIR=/path/to/external/data            # External data directory to mount
```

### Manual Installation

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install transformers==4.21.0 \
            pytorch-lightning==1.7.4 \
            einops==0.6.0 \
            pytest==7.2.1 \
            sentence-transformers==2.2.2 \
            faiss-cpu==1.7.4 \
            matplotlib==3.6.3 \
            seaborn==0.12.2
```

For GPU-accelerated FAISS, replace `faiss-cpu` with `faiss-gpu`.

### Pre-trained Model

Download the [pre-trained model](https://drive.google.com/file/d/1R53kuW_6uWDCXamQy2AghgeseaIPsAcB/view?usp=sharing) and extract it into `data/model/`.

## Data Preparation Pipeline

Building a RETRO dataset involves a multi-step pipeline. See [`data/datasets/README.md`](data/datasets/README.md) for full details.

Input data should be in JSONL format with one document per line:

```json
{"text": "Here is document 1"}
{"text": "Here is document 2"}
```

### 1. Tokenize and Chunk

Split documents into fixed-size token chunks:

```bash
cat data.jsonl | python src/data/tokenize_and_chunk.py \
    --chunks-out chunks.npy \
    --seq2chunk-index-out seq2chunk.npy \
    --chunk2seq-index-out chunk2seq.npy \
    --chunk-size 64 \
    --max-chunks 10000000
```

**Outputs:** `chunks.npy` (tokenized chunks), `seq2chunk.npy` (document-to-chunk mapping), `chunk2seq.npy` (chunk-to-document mapping).

### 2. Embed Chunks

Generate Sentence Transformer embeddings for similarity search:

```bash
python src/data/embed_chunks.py \
    chunks.npy \
    chunks.embeddings.npy \
    --batch-size 256
```

Then create an index spec file (`index.spec.json`) listing all shards:

```json
[
    {
        "chunks": "chunks.npy",
        "seq2chunk": "seq2chunk.npy",
        "chunk2seq": "chunk2seq.npy",
        "embeddings": "chunks.embeddings.npy"
    }
]
```

### 3. Train FAISS Index

```bash
python src/data/train_faiss_index.py \
    --spec index.spec.json \
    --max-training-vectors $((131072 * 256)) \
    --index-type IVF131072,PQ32 \
    --output trained.index \
    --use-gpus  # optional
```

### 4. Build Index

Populate the trained index with chunk embeddings:

```bash
python src/data/build_faiss_index.py \
    --spec index.spec.json \
    --trained-index trained.index \
    --output-index data.index \
    --use-gpus       # optional
    --shard-index    # optional, for multi-GPU
```

### 5. Retrieve Neighbors

Pre-compute nearest neighbors for each chunk:

```bash
python src/data/retrieve_neighbours.py \
    --query-embeddings chunks.embeddings.npy \
    --query-chunk2seq chunk2seq.npy \
    --neighbours-output chunks.neighbours.npy \
    --index data.index \
    --index-spec index.spec.json \
    --num-neighbours 4
```

### 6. Create Dataset Spec

Create a `data.spec.json` file that references all the prepared data:

```json
{
    "shards": [
        {
            "chunks": "chunks.npy",
            "seq2chunk": "seq2chunk.npy",
            "chunk2seq": "chunk2seq.npy",
            "neighbours": "chunks.neighbours.npy"
        }
    ],
    "neighbours": {
        "faiss_index": "data.index",
        "index_spec": "index.spec.json"
    }
}
```

This spec file is used as input for training, evaluation, and generation.

## Training

Train a RETRO model using PyTorch Lightning:

```bash
python src/train_retro.py \
    --training-dataset-spec /path/to/train.spec.json \
    --validation-dataset-spec /path/to/val.spec.json \
    --retro-config /path/to/config.json \
    --batch-size 32 \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
    --experiment-dir /path/to/experiment \
    --gpus-per-node 1
```

### Training Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--training-dataset-spec` | Yes | - | Path to training dataset spec JSON |
| `--validation-dataset-spec` | Yes | - | Path to validation dataset spec JSON |
| `--retro-config` | Yes | - | Path to model configuration JSON |
| `--batch-size` | Yes | - | Batch size per GPU |
| `--num-neighbours` | No | None | Number of retrieved neighbors per chunk |
| `--num-continuation-chunks` | No | 1 | Number of continuation chunks per neighbor |
| `--max-len` | No | None | Maximum sequence length |
| `--experiment-dir` | No | None | Directory for checkpoints and TensorBoard logs |
| `--gpus-per-node` | No | None | Number of GPUs per node |
| `--num-nodes` | No | None | Number of nodes for distributed training |
| `--accumulate-grad-batches` | No | None | Gradient accumulation steps |
| `--val-check-interval` | No | 20000 | Validation frequency (in training steps) |
| `--training-data-subset-indices` | No | None | File with subset indices for training data |
| `--validation-data-subset-indices` | No | None | File with subset indices for validation data |

Training uses AdamW optimizer (lr=1e-4, weight_decay=0.1) with gradient clipping at 1.0.

## Evaluation

Evaluate a trained model on test data:

```bash
python src/evaluate_retro.py \
    --test-dataset-spec /path/to/test.spec.json \
    --checkpoint /path/to/checkpoint.ckpt \
    --retro-config /path/to/config.json \
    --batch-size 32 \
    --num-neighbours 2
```

## Text Generation

Generate text interactively with a trained model:

```bash
python src/generate_retro.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --retro-config /path/to/config.json \
    --num-neighbours 2 \
    --prompt "Your prompt here"
```

Omit `--prompt` to enter an interactive loop where you provide prompts and neighbor text at runtime. Generation uses greedy decoding by default with `no_repeat_ngram_size=3` and a maximum length of 200 tokens.

### Index Service

For production-style retrieval during generation, a FastAPI-based index service can serve nearest-neighbor queries over HTTP:

```bash
# Start the FAISS HTTP server
python src/faiss_http_server.py

# Start the index service
python src/index_service.py
```

## Testing

Run the test suite with pytest:

```bash
pytest src/test_modeling_retro.py
```

Tests cover:
- **Batch independence** - Verifies that gradients for one example in a batch don't affect other examples
- **Causality** - Ensures the model output at position `t` doesn't depend on inputs at positions `> t`
- **Encoder-decoder integration** - Validates the full forward pass through encoder and decoder
- **Attention mechanisms** - Tests self-attention and cross-attention modules

## Configuration Reference

The model is configured via a JSON file passed as `--retro-config`. All fields correspond to the `RetroConfig` dataclass:

```json
{
    "num_embeddings": 28996,
    "pad_token_idx": 0,
    "chunk_size": 64,
    "dropout_p": 0.1,

    "enc_hidden_dim": 768,
    "enc_num_layers": 2,
    "enc_ffn_dim": 3072,
    "enc_num_heads": 4,
    "enc_qkv_dim": 768,
    "enc_ca_layers": [0, 1],

    "dec_hidden_dim": 768,
    "dec_num_layers": 2,
    "dec_ffn_dim": 3072,
    "dec_num_heads": 4,
    "dec_qkv_dim": 768,
    "dec_cca_layers": [1]
}
```

| Parameter | Default | Description |
|---|---|---|
| `num_embeddings` | 28996 | Vocabulary size |
| `pad_token_idx` | 0 | Padding token index |
| `chunk_size` | 64 | Number of tokens per chunk |
| `dropout_p` | 0.1 | Dropout probability |
| `enc_hidden_dim` | 768 | Encoder hidden dimension |
| `enc_num_layers` | 2 | Number of encoder layers |
| `enc_ffn_dim` | 3072 | Encoder feed-forward dimension |
| `enc_num_heads` | 4 | Encoder attention heads |
| `enc_qkv_dim` | 768 | Encoder query/key/value dimension |
| `enc_ca_layers` | [0, 1] | Encoder layers with cross-attention |
| `dec_hidden_dim` | 768 | Decoder hidden dimension |
| `dec_num_layers` | 2 | Number of decoder layers |
| `dec_ffn_dim` | 3072 | Decoder feed-forward dimension |
| `dec_num_heads` | 4 | Decoder attention heads |
| `dec_qkv_dim` | 768 | Decoder query/key/value dimension |
| `dec_cca_layers` | [1] | Decoder layers with chunked cross-attention |

## Citation

This implementation is based on:

```bibtex
@article{borgeaud2021improving,
    title={Improving Language Models by Retrieving from Trillions of Tokens},
    author={Borgeaud, Sebastian and Mensch, Arthur and Hoffmann, Jordan and
            Cai, Trevor and Rutherford, Eliza and Millican, Katie and
            van den Driessche, George Bm and Lespiau, Jean-Baptiste and
            Damoc, Bogdan and Clark, Aidan and others},
    journal={arXiv preprint arXiv:2112.05329},
    year={2021}
}
```
