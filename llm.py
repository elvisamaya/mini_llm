from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DataModule:
    def __init__(self, text: str):
        self.tokenizer = CharTokenizer(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        split = int(0.9 * len(data))
        self.train_data = data[:split]
        self.val_data = data[split:]

    def get_batch(self, split: str, block_size: int, batch_size: int):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y


class MiniLLM(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(t))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    text = Path("data.txt").read_text(encoding="utf-8")
    data_module = DataModule(text)

    block_size = 32
    batch_size = 32
    n_embd = 64
    max_iters = 1200

    model = MiniLLM(data_module.tokenizer.vocab_size, block_size, n_embd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(max_iters):
        xb, yb = data_module.get_batch("train", block_size, batch_size)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    context = torch.zeros((1, 1), dtype=torch.long)
    output = model.generate(context, max_new_tokens=300)[0].tolist()

    print("\n=== SAMPLE OUTPUT ===\n")
    print(data_module.tokenizer.decode(output))


if __name__ == "__main__":
    main()
