# MiniTransformer

A minimal character-level GPT implementation built from scratch using PyTorch. This transformer model uses multi-head self-attention, residual connections, and layer normalization to generate text after training on a single dataset file. Features 8 transformer blocks, 8 attention heads, 256-dimensional embeddings, and a 128-token context window. Supports both training and interactive chat modes.

## Usage

```bash
# Install PyTorch
pip install torch

# Run the chat interface
python PersonalGPT.py
```

The model loads pre-trained weights from `personalgpt.pth` and starts an interactive session. Type `exit` to quit.

## Training

To train on custom data, replace `astronomy.txt` with your own text file (minimum 500 characters), uncomment the training loop in `PersonalGPT.py`, and run the script. The model will train for 3000 iterations and save weights to `personalgpt.pth`.

## Architecture

Built with standard transformer components: token and positional embeddings, multi-head self-attention with causal masking, feed-forward networks with ReLU activation, and layer normalization. Uses character-level tokenization for simplicity. Implements autoregressive generation by sampling from the output distribution at each step.
