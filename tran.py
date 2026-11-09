import math
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

amp_enabled = device != "cuda" or torch.cuda.is_bf16_supported()

from model import NanoGPTLModel
from prepare import get_batch, tokenizer

# ---------- config ----------
max_iters = 5000
eval_interval = 100
eval_iters = 100
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 200
weight_decay = 0.1
patience = 15 

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_FILE = WEIGHTS_DIR / "model.pt"

model = NanoGPTLModel().to(device)
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

raw_model = model
if device == "cuda":
    model = torch.compile(model)

decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=learning_rate,
)

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp_enabled):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

print("เริ่มการเทรนน้อง NanoGPTLModeloGPT")
best_val = float("inf")
bad_evals = 0

for it in range(max_iters):
    for g in optimizer.param_groups:
        g["lr"] = get_lr(it)

    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
              f"gap {losses['val'] - losses['train']:.4f}, lr {get_lr(it):.2e}")
        if losses["val"] < best_val:
            best_val = losses["val"]
            bad_evals = 0
            torch.save(raw_model.state_dict(), WEIGHTS_FILE)
        else:
            bad_evals += 1
            if bad_evals >= patience:
                print(f"Early stopping ที่ step {it} (best val {best_val:.4f})")
                break

    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp_enabled):
        _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print(f"เสร็จแล้ว best val loss = {best_val:.4f} (บันทึกไว้ที่ {WEIGHTS_FILE})")

raw_model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
raw_model.eval()
context = torch.zeros((1, 1), dtype=torch.long).to(device)
out = raw_model.generate(context, max_new_token=100, temperature=0.8, top_k=50)
print("==== ข้อความที่โมเดลสร้าง ====")
print(tokenizer.decode(out[0].tolist()))
