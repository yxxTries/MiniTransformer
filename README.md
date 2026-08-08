# MiniTransformer

A minimal GPT-style language model built from a small section of shakesperean text.

## Features

- Character-level transformer with multi-head self-attention
- Token and positional embeddings
- Configurable architecture (layers, heads, embedding size)
- Interactive chat mode

## Requirements

- Python 3.x
- PyTorch

```bash
pip install torch
```

## Usage

**Chat with the model:**
```bash
python PersonalGPT.py
```

**Train from scratch:**  
Uncomment the training loop in `PersonalGPT.py` and run `python PersonalGPT.py`.

## Architecture

| Component | Default |
|-----------|---------|
| Embedding dim | 256 |
| Attention heads | 8 |
| Transformer layers | 8 |
| Context length | 128 |

## Files

- `PersonalGPT.py` — Model definition and chat interface
- `astronomy.txt` — Training dataset
- `personalgpt.pth` — Pre-trained weights
