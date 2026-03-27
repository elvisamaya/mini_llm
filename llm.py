import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class Trainer:
    def __init__(self, text: str, device: str = "cpu"):
        self.tokenizer = CharTokenizer(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        split = int(0.9 * len(data))
        self.train_data = data[:split]
        self.val_data = data[split:]
        self.device = device

    def get_batch(self, split: str, batch_size: int, block_size: int):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self, model: MiniLLM, eval_iters: int, batch_size: int, block_size: int):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = self.get_batch(split, batch_size, block_size)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out


def train_model(
    input_file: str,
    out_file: str,
    max_iters: int = 3000,
    eval_interval: int = 300,
    eval_iters: int = 50,
    learning_rate: float = 3e-4,
    batch_size: int = 32,
    block_size: int = 128,
    n_embd: int = 128,
    n_head: int = 4,
    n_layer: int = 4,
    dropout: float = 0.2,
    generate_tokens: int = 500,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = Path(input_file).read_text(encoding="utf-8")
    trainer = Trainer(text, device=device)

    model = MiniLLM(
        vocab_size=trainer.tokenizer.vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = trainer.estimate_loss(model, eval_iters, batch_size, block_size)
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = trainer.get_batch("train", batch_size, block_size)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": trainer.tokenizer.stoi,
            "itos": trainer.tokenizer.itos,
            "config": {
                "vocab_size": trainer.tokenizer.vocab_size,
                "block_size": block_size,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "dropout": dropout,
            },
        },
        out_file,
    )

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=generate_tokens)[0].tolist()

    print("\n=== SAMPLE OUTPUT ===\n")
    print(trainer.tokenizer.decode(generated))


def sample_model(model_file: str, prompt: str, max_new_tokens: int = 300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_file, map_location=device)

    model = MiniLLM(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s if c in stoi]

    def decode(ids: list[int]) -> str:
        return "".join(itos[i] for i in ids)

    start_ids = encode(prompt)
    if not start_ids:
        start_ids = [0]

    idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    print(decode(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or sample a mini character-level LLM.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input", required=True, help="Path to a training text file")
    train_parser.add_argument("--out", default="mini_llm.pt", help="Where to save the trained model")
    train_parser.add_argument("--max-iters", type=int, default=3000)
    train_parser.add_argument("--eval-interval", type=int, default=300)
    train_parser.add_argument("--eval-iters", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--block-size", type=int, default=128)
    train_parser.add_argument("--n-embd", type=int, default=128)
    train_parser.add_argument("--n-head", type=int, default=4)
    train_parser.add_argument("--n-layer", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--generate-tokens", type=int, default=500)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--model", required=True, help="Path to a saved .pt model")
    sample_parser.add_argument("--prompt", default="Hello", help="Prompt to start generation")
    sample_parser.add_argument("--max-new-tokens", type=int, default=300)

    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            input_file=args.input,
            out_file=args.out,
            max_iters=args.max_iters,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
            generate_tokens=args.generate_tokens,
        )
    elif args.mode == "sample":
        sample_model(
            model_file=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
