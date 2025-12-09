# src/train_tpse_ssl.py
"""
B-Lite: Optimized Self-Supervised TPSE pretraining (stable)
Improvements:
 - LayerNorm in model (tpse_model.py) to stabilize scale
 - contrastive proj vectors normalized before NT-Xent
 - higher temperature (0.2)
 - recon MSE uses 'mean'
 - gradient clipping
 - guarded/optional order+mask tasks
"""

import os
import random
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tpse_dataset import TPSEDataset, collate_fn
from tpse_model import TPSEModel  # uses updated TPSEModel with LayerNorm

# -----------------------
# CONFIG
# -----------------------
SEED = 42
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-5
MASK_PROB = 0.15
CONTRAST_TEMP = 0.2
CONTRAST_WEIGHT = 1.0
ORDER_WEIGHT = 0.5
MASK_WEIGHT = 1.0
GRAD_CLIP = 1.0

ORDER_EVERY_N = 3
MASK_EVERY_N = 2

CACHE_DIR = Path("data_processed/tpse_cache")
LABELS_PATH = Path("data_processed/intermediate/labels.csv")
CHECKPOINT_IN = Path("models/tpse_model.pt")
CHECKPOINT_OUT = Path("models/tpse_ssl.pt")

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# Utilities
# -----------------------
def load_labels_dict(path):
    if not path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(path, dtype=str)
    if "MRN" in df.columns and "mrn" not in df.columns:
        df = df.rename(columns={"MRN": "mrn"})
    df["mrn"] = df["mrn"].astype(str).str.strip()
    if "complication_flag" in df.columns:
        df["complication_flag"] = df["complication_flag"].astype(int)
    else:
        df["complication_flag"] = 0
    return dict(zip(df["mrn"].tolist(), df["complication_flag"].tolist()))

def make_mask_from_lengths(lengths, max_len=None, device=None):
    if isinstance(lengths, (list, tuple)):
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    if max_len is None:
        max_len = int(lengths.max().item())
    rng = torch.arange(max_len, device=device).unsqueeze(0)
    return rng < lengths.unsqueeze(1)

# -----------------------
# Augmentations
# -----------------------
def augment_jitter_dropout_mask(seq, lengths, jitter_std=0.01, drop_rate=0.05, mask_rate=0.03):
    """
    seq: (B, T, D)
    lengths: (B,)
    """
    B, T, D = seq.shape
    seq = seq.clone()
    for i in range(B):
        L = int(lengths[i].item()) if hasattr(lengths[i], 'item') else int(lengths[i])
        if L <= 0:
            continue
        # jitter (per timestep)
        noise = torch.randn((L, D), device=seq.device) * jitter_std
        seq[i, :L] = seq[i, :L] + noise
        # time-dropout: drop whole timesteps
        if drop_rate > 0:
            num_drop = max(1, int(L * drop_rate))
            drop_idx = torch.randperm(L, device=seq.device)[:num_drop]
            seq[i, drop_idx] = 0.0
        # feature mask (mask some features across valid timesteps)
        if mask_rate > 0:
            num_mask = max(1, int(D * mask_rate))
            feat_idx = torch.randperm(D, device=seq.device)[:num_mask]
            seq[i, :L, feat_idx] = 0.0
    return seq

# -----------------------
# Sampling for order task
# -----------------------
def _ensure_2d_tensor(x):
    if torch.is_tensor(x):
        if x.ndim == 1:
            return x.unsqueeze(0)
        return x
    arr = np.asarray(x)
    if arr.ndim == 1:
        return torch.tensor(arr).unsqueeze(0)
    return torch.tensor(arr)

def sample_two_subsegments_single(seq, seg_len=4):
    seq = _ensure_2d_tensor(seq)
    T, D = seq.shape
    min_needed = seg_len * 2
    if T < min_needed:
        pad_n = min_needed - T
        pad = seq[-1:].repeat(pad_n, 1)
        seq = torch.cat([seq, pad], dim=0)
        T = seq.shape[0]
    i1 = random.randint(0, T - seg_len)
    i2 = random.randint(0, T - seg_len)
    A = seq[i1:i1+seg_len]
    B = seq[i2:i2+seg_len]
    if random.random() < 0.5:
        return torch.stack([A, B]), 1
    else:
        return torch.stack([B, A]), 0

def sample_pairs_for_batch(seq_batch, length_batch, seg_len=4, device="cpu"):
    B, T, D = seq_batch.shape
    pairs, labels = [], []
    for i in range(B):
        L = int(length_batch[i].item()) if hasattr(length_batch[i], 'item') else int(length_batch[i])
        s = seq_batch[i, :max(1, L)].cpu()
        p, lab = sample_two_subsegments_single(s, seg_len)
        pairs.append(p)
        labels.append(lab)
    pair_tensor = torch.stack(pairs, dim=0).to(device)
    order_labels = torch.tensor(labels, dtype=torch.long, device=device)
    return pair_tensor, order_labels

# -----------------------
# Contrastive loss (NT-Xent)
# -----------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        B = z1.shape[0]
        if B == 0:
            return torch.tensor(0.0, device=z1.device)
        z = torch.cat([z1, z2], dim=0)  # (2B, dim)
        sim = torch.matmul(z, z.T) / self.temperature
        diag = torch.eye(2 * B, device=sim.device).bool()
        sim_masked = sim.masked_fill(diag, -9e15)
        positives = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)
        lse = torch.logsumexp(sim_masked, dim=1)
        loss = (-positives + lse).mean()
        return loss

# -----------------------
# SSL Heads
# -----------------------
class SSLHeads(nn.Module):
    def __init__(self, hidden_dim, input_dim, proj_dim=128, order_hidden=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.order_clf = nn.Sequential(
            nn.Linear(hidden_dim * 2, order_hidden),
            nn.ReLU(),
            nn.Linear(order_hidden, 2)
        )
        self.recon = nn.Linear(hidden_dim, input_dim)

# -----------------------
# Dataloader builder
# -----------------------
def build_dataloader(batch_size):
    files = sorted(CACHE_DIR.glob("*.npz"))
    if len(files) == 0:
        raise RuntimeError("No TPSE cache found. Run tpse_dataset_builder first.")
    mrn_list = [p.stem for p in files]
    labels_dict = load_labels_dict(LABELS_PATH)
    ds = TPSEDataset(mrn_list, labels_dict, str(CACHE_DIR))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

# -----------------------
# Training loop
# -----------------------
def train():
    loader = build_dataloader(BATCH_SIZE)

    if not CHECKPOINT_IN.exists():
        raise FileNotFoundError("Missing TPSE checkpoint at " + str(CHECKPOINT_IN))

    ckpt = torch.load(CHECKPOINT_IN, map_location="cpu")

    sample_npz = next(CACHE_DIR.glob("*.npz"))
    sample = np.load(sample_npz)
    input_dim = int(sample["seq"].shape[1])

    model = TPSEModel(input_dim=input_dim, hidden_dim=128, n_layers=2, n_heads=4, dropout=0.1)
    # load existing keys; input_norm may be new so use strict=False
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print("Loaded checkpoint with strict=False (ignored missing keys)")
    model.to(DEVICE)

    heads = SSLHeads(hidden_dim=128, input_dim=input_dim).to(DEVICE)

    contrastive_loss = NTXentLoss(CONTRAST_TEMP)
    recon_loss_fn = nn.MSELoss(reduction="mean")   # stable mean
    order_loss_fn = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(list(model.parameters()) + list(heads.parameters()), lr=LR)

    global_step = 0
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train(); heads.train()
            total_loss = 0.0; n_batches = 0

            for batch in loader:
                seq, static, label, lengths = batch
                seq = seq.to(DEVICE)
                lengths = lengths.to(DEVICE)
                B, T, D = seq.shape
                if B == 0:
                    continue

                # --- contrastive two views ---
                v1 = augment_jitter_dropout_mask(seq, lengths, jitter_std=0.01, drop_rate=0.05, mask_rate=0.03)
                v2 = augment_jitter_dropout_mask(seq, lengths, jitter_std=0.02, drop_rate=0.08, mask_rate=0.05)

                mask = make_mask_from_lengths(lengths, T, DEVICE)
                o1 = model(v1, mask)
                o2 = model(v2, mask)

                mask_float = mask.float().unsqueeze(-1)
                p1 = (o1 * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
                p2 = (o2 * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)

                z1 = heads.proj(p1)
                z2 = heads.proj(p2)
                z1 = nn.functional.normalize(z1, dim=-1)
                z2 = nn.functional.normalize(z2, dim=-1)

                loss_contrast = contrastive_loss(z1, z2)

                # --- temporal order (occasionally) ---
                loss_order = torch.tensor(0.0, device=DEVICE)
                if (global_step % ORDER_EVERY_N) == 0:
                    pair, order_labels = sample_pairs_for_batch(seq, lengths, seg_len=4, device=DEVICE)
                    if pair is not None:
                        Bp, _, Ls, D2 = pair.shape
                        flat = pair.view(Bp * 2, Ls, D2)
                        flat_len = torch.tensor([Ls] * (Bp * 2), device=DEVICE)
                        out = model(flat, make_mask_from_lengths(flat_len, Ls, DEVICE))
                        pooled = out.mean(dim=1).view(Bp, 2, -1)
                        concat = torch.cat([pooled[:, 0], pooled[:, 1]], dim=-1)
                        logits = heads.order_clf(concat)
                        loss_order = order_loss_fn(logits, order_labels)

                # --- masked reconstruction (occasionally) ---
                loss_mask = torch.tensor(0.0, device=DEVICE)
                if (global_step % MASK_EVERY_N) == 0:
                    seq_masked = seq.clone()
                    maskpos = torch.zeros((B, T), dtype=torch.bool, device=DEVICE)
                    for i in range(B):
                        L = int(lengths[i].item())
                        k = max(1, int(L * MASK_PROB))
                        perm = torch.randperm(L, device=DEVICE)[:k]
                        seq_masked[i, perm] = 0.0
                        maskpos[i, perm] = True

                    outm = model(seq_masked, mask)
                    # normalize per-timestep representations (stabilizer)
                    norm = outm.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    outm_norm = outm / norm

                    pred = heads.recon(outm_norm)  # (B, T, D)
                    mp = maskpos.unsqueeze(-1).expand(-1, -1, D)
                    if mp.any():
                        t_pred = pred[mp].view(-1, D)
                        t_true = seq[mp].view(-1, D)
                        loss_mask = recon_loss_fn(t_pred, t_true)

                loss = (CONTRAST_WEIGHT * loss_contrast +
                        ORDER_WEIGHT * loss_order +
                        MASK_WEIGHT * loss_mask)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), GRAD_CLIP)
                optim.step()

                total_loss += float(loss.item())
                n_batches += 1
                global_step += 1

                if global_step % 50 == 0:
                    print(f"Epoch {epoch} step {global_step} avg_loss {total_loss / max(1,n_batches):.6f}")

            print(f"Epoch {epoch} completed. avg_loss={total_loss / max(1,n_batches):.6f}")
            torch.save({"model_state": model.state_dict(),
                        "heads_state": heads.state_dict(),
                        "optim_state": optim.state_dict(),
                        "epoch": epoch}, CHECKPOINT_OUT)

    except KeyboardInterrupt:
        print("KeyboardInterrupt — saving partial checkpoint.")
        torch.save({"model_state": model.state_dict(),
                    "heads_state": heads.state_dict(),
                    "optim_state": optim.state_dict(),
                    "epoch": "interrupted"}, CHECKPOINT_OUT)
        raise

    print("SSL TRAINING COMPLETE — saved:", CHECKPOINT_OUT)


if __name__ == "__main__":
    train()
