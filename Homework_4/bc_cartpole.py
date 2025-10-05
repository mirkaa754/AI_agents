# bc_cartpole.py  — Behavior Cloning on CartPole from an HF dataset
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split

HF_DATASET_ID = "NathanGavenski/CartPole-v1"  # change if you want a different dataset
SUBSET = 150_000  # set smaller subset to speed up training; None to use full

# ---- helpers to robustly find columns ----
def pick_first_present(ds, candidates):
    cols = set(ds.column_names)
    for c in candidates:
        if c in cols:
            return c
    return None

def to_numpy_array(col):
    # Handles list/array columns of variable types
    arr = np.array(col)
    if arr.dtype == object:
        # list of lists -> stack
        arr = np.stack([np.array(x, dtype=np.float32) for x in col])
    return arr

# ---- load and map columns ----
print(f"Loading dataset: {HF_DATASET_ID}")
ds = load_dataset(HF_DATASET_ID, split="train")
print("Columns:", ds.column_names)
# Try to locate observation and action columns
obs_col = pick_first_present(ds, ["obs", "observation", "observations", "state", "states", "x", "features"])
act_col = pick_first_present(ds, ["action", "actions", "a", "act", "label", "labels", "teacher_action"])
logits_col = pick_first_present(ds, ["logits", "policy_logits", "teacher_logits", "pi_logits", "logits_teacher"])

if obs_col is None:
    raise ValueError(f"Could not find an observation column. Available: {ds.column_names}")

X = to_numpy_array(ds[obs_col]).astype(np.float32)

# Derive actions:
if act_col is not None:
    y = np.array(ds[act_col]).astype(np.int64)
elif logits_col is not None:
    # Derive discrete actions from class logits/probs
    raw = ds[logits_col]
    # raw could be list[list]; make array and argmax
    y = np.argmax(np.array(raw, dtype=np.float32), axis=-1).astype(np.int64)
else:
    raise ValueError(
        "Could not find an action column ('action','actions','act','label',...) "
        "or a logits column to derive actions."
    )

# Optional: subsample for faster training
if SUBSET is not None and SUBSET < len(X):
    idx = np.random.default_rng(0).choice(len(X), size=SUBSET, replace=False)
    X, y = X[idx], y[idx]

# ---- simple MLP policy ----
class MLP(nn.Module):
    def __init__(self, in_dim=4, hid=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x):
        return self.net(x)

in_dim = X.shape[1]
n_actions = int(np.max(y)) + 1  # infer #classes from labels
print(f"in_dim={in_dim}, n_actions={n_actions}, samples={len(X)}")

# Train/val split
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

device = "cuda" if th.cuda.is_available() else "cpu"
policy = MLP(in_dim=in_dim, hid=128, out_dim=n_actions).to(device)
opt = optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def batches(X, y, bs=256):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    for i in range(0, len(X), bs):
        idb = idx[i:i+bs]
        yield th.tensor(X[idb]).to(device), th.tensor(y[idb]).to(device)

EPOCHS = 15
for epoch in range(EPOCHS):
    policy.train()
    tr_loss = 0.0
    for xb, yb in batches(Xtr, ytr, bs=256):
        opt.zero_grad()
        logits = policy(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        tr_loss += float(loss.item())
    policy.eval()
    with th.no_grad():
        xv = th.tensor(Xval).to(device)
        yv = th.tensor(yval).to(device)
        pred = policy(xv).argmax(-1)
        acc = (pred == yv).float().mean().item()
    print(f"epoch {epoch+1:02d} | train_loss {tr_loss:.2f} | val_acc {acc:.3f}")

os.makedirs("models", exist_ok=True)
th.save({"state_dict": policy.state_dict(), "in_dim": in_dim, "n_actions": n_actions}, "models/bc_cartpole_policy.pt")
print("Saved -> models/bc_cartpole_policy.pt")
