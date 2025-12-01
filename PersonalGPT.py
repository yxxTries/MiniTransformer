import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------
# Hyperparameters
# -------------------
batch_size = 16        # sequences per batch
block_size = 128       # max context length
n_embd = 256           # embedding dimension
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# -------------------
# Load dataset from text file
# -------------------
dataset_path = "astronomy.txt"

try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find {dataset_path}. Make sure it is in the right destination."
    )

if len(text) < 500:
    raise ValueError(
        f"Dataset too small (len={len(text)} chars). Add more text to {dataset_path}."
    )

print("Loaded dataset with", len(text), "characters")

# -------------------
# Build vocabulary
# -------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return ''.join([itos[i] for i in indices])

# Encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Vocab size:", vocab_size)
print("Train tokens:", len(train_data))
print("Val tokens:", len(val_data))

# -------------------
# Batch loader
# -------------------
def get_batch(split):
    """Generate a batch of (x, y) sequences."""
    data_split = train_data if split == 'train' else val_data
    max_start = len(data_split) - block_size - 1

    if max_start <= 0:
        raise ValueError(
            f"Dataset too small for block_size={block_size}. "
            f"Reduce block_size or add more text to astronomy.txt."
        )

    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# -------------------
# Token + Positional Embeddings
# -------------------
class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        B, T = idx.shape
        token_embeddings = self.token_emb(idx)            # (B, T, C)
        positions = torch.arange(T, device=idx.device)    # (T,)
        pos_embeddings = self.pos_emb(positions)          # (T, C)
        pos_embeddings = pos_embeddings.unsqueeze(0)      # (1, T, C)
        return token_embeddings + pos_embeddings          # (B, T, C)


class SelfAttentionHead(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # compute Q, K, V
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # attention scores
        att = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)

        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v      # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead(head_size, n_embd, block_size, dropout)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """A simple MLP used inside the transformer block."""
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One full Transformer block: LayerNorm → MHA → Add → LayerNorm → MLP → Add"""

    def __init__(self, n_embd, num_heads, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        self.mha = MultiHeadAttention(num_heads, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        # 1. Self-attention block
        x = x + self.mha(self.ln1(x))  # (residual connection)

        # 2. Feed-forward block
        x = x + self.ffwd(self.ln2(x))  # (residual connection)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads=4, n_layers=2, dropout=0.0):
        super().__init__()

        # 1) Token + positional embeddings
        self.tok_pos_emb = TokenAndPositionalEmbedding(vocab_size, n_embd, block_size)

        # 2) Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, num_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        # 3) Final LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        # 4) Output projection to vocabulary logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size

    def forward(self, idx, targets=None):
        """
        idx: (B, T) input token indices
        targets: (B, T) target token indices or None

        returns:
            logits: (B, T, vocab_size)
            loss: scalar (if targets is not None), else None
        """
        B, T = idx.shape

        # Safety: don't feed sequences longer than block_size
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")

        # 1) Embed tokens + positions
        x = self.tok_pos_emb(idx)          # (B, T, C)

        # 2) Pass through each transformer block
        for block in self.blocks:
            x = block(x)                   # still (B, T, C)

        # 3) Final layer norm
        x = self.ln_f(x)                   # (B, T, C)

        # 4) Project to vocabulary logits
        logits = self.lm_head(x)           # (B, T, vocab_size)

        # If no targets given: just return logits (e.g. for generation)
        if targets is None:
            return logits, None

        # For training: compute cross-entropy loss
        # Flatten batch & time so we get shape (B*T, vocab_size)
        B, T, V = logits.shape
        logits_flat = logits.view(B * T, V)
        targets_flat = targets.view(B * T)

        loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generate new tokens.
        idx: (B, T) current context
        returns: (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # Get logits for current context
            logits, _ = self(idx_cond)           # (B, T, vocab_size)

            # Take last time step
            logits_last = logits[:, -1, :]       # (B, vocab_size)

            # Convert to probabilities
            probs = F.softmax(logits_last, dim=-1)  # (B, vocab_size)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            idx = torch.cat((idx, next_token), dim=1)  # (B, T+1)

        return idx

#chat function
def chat(model):
    model.eval()
    print("\n=== MiniGPT Interactive Chat ===")
    print("Type 'exit' to quit.\n")

    # Start with an empty context
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    while True:
        # User input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Encode user text → tokens
        user_tokens = torch.tensor(
            encode(user_input),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)  # (1, T)

        # Append user tokens to the running context
        context = torch.cat((context, user_tokens), dim=1)

        # Generate a continuation from the model
        output = model.generate(context, max_new_tokens=200)

        # Decode everything so far
        full_text = decode(output[0].tolist())

        # Extract only the NEW part (the bot's answer)
        bot_answer = full_text[len(decode(context[0].tolist())):]

        # If empty (rare), try generating again
        if bot_answer.strip() == "":
            bot_answer = "(no response generated)"

        print("MiniGPT:", bot_answer)

        # Add bot response tokens to context (so it remembers)
        context = torch.tensor(
            encode(decode(context[0].tolist()) + bot_answer),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

if __name__ == "__main__":
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        num_heads=8,
        n_layers=8,
        dropout=0.1
    ).to(device)

    model.load_state_dict(torch.load("Personalgpt.pth"))  # Load trained weights
    
    chat(model)


# #Training Loop
# if __name__ == "__main__":

#     model = GPTLanguageModel(
#         vocab_size=vocab_size,
#         n_embd=n_embd,
#         block_size=block_size,
#         num_heads=8,
#         n_layers=8,
#         dropout=0.1,
#     ).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

#     max_iters = 3000
#     eval_interval = 300
#     eval_iters = 200

#     print("Training starting...")

#     for step in range(max_iters):

#         # ---- TRAIN STEP ----
#         xb, yb = get_batch('train')
#         logits, loss = model(xb, yb)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # ---- PERIODIC EVALUATION ----
#         if step % eval_interval == 0:
#             model.eval()

#             with torch.no_grad():
#                 losses = []
#                 for _ in range(eval_iters):
#                     xb, yb = get_batch('val')
#                     _, val_loss = model(xb, yb)
#                     losses.append(val_loss.item())

#             avg_val_loss = sum(losses) / len(losses)
#             print(f"Step {step}/{max_iters} | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f}")

#             model.train()

#     print("\nTraining complete!")

#     # ---- SAVE MODEL ----
#     torch.save(model.state_dict(), "personalgpt.pth")
#     print("Model saved as minigpt.pth")

#     # ---- GENERATE SAMPLE TEXT ----
#     print("\nGenerating sample text:")
#     model.eval()

#     context = torch.zeros((1, 1), dtype=torch.long, device=device)
#     output = model.generate(context, max_new_tokens=400)

#     print(decode(output[0].tolist()))



#------------------------------------------------------------- Testing Components -------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------------------------------------------|

# # Test batch creation
# xb, yb = get_batch('train')
# print("Batch shapes:", xb.shape, yb.shape)

# # Test embeddings
# emb = TokenAndPositionalEmbedding(vocab_size, n_embd, block_size).to(device)
# xemb = emb(xb)
# print("Embedding output:", xemb.shape)  # (B, T, C)

# #Test Multi-Head Attention
# mha = MultiHeadAttention(num_heads=4, n_embd=n_embd, block_size=block_size).to(device)
# out2 = mha(xemb)
# print("Multi-Head output:", out2.shape)

# # Test Transformer Block
# block = TransformerBlock(n_embd, num_heads=4, block_size=block_size).to(device)
# out3 = block(xemb)
# print("Transformer block output:", out3.shape)

# # Test GPT Language Model
# if __name__ == "__main__":
#     # Get a batch
#     xb, yb = get_batch('train')  # (B, T)

#     # Create model
#     model = GPTLanguageModel(
#         vocab_size=vocab_size,
#         n_embd=n_embd,
#         block_size=block_size,
#         num_heads=4,
#         n_layers=2,
#         dropout=0.0
#     ).to(device)

#     # Forward pass
#     logits, loss = model(xb, yb)
#     print("Logits shape:", logits.shape)  # (B, T, vocab_size)
#     print("Loss:", loss.item())

#     # Try generating text (untrained, so it'll be gibberish but structurally correct)
#     context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with token 0
#     generated = model.generate(context, max_new_tokens=200)
#     print("\nSample output:")
#     print(decode(generated[0].tolist()))