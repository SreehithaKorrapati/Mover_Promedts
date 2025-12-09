# src/train_mtif_with_cot.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

# import your model and the loader you've been using
from src.mtif_model import MTIFModel
from src.train_mtif_baseline import load_and_align

# -------------------------
# Helpers
# -------------------------
def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute classification and medical metrics robustly.
    y_true: numpy array (N,)
    y_prob: numpy array (N,)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # AUC (may fail if only one class present)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    # Binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Precision, recall, f1 (binary). zero_division=0 to avoid exceptions.
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # Confusion matrix in a safe way
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except Exception:
        # If confusion_matrix can't return 4 values (e.g. only one class present)
        tn = fp = fn = tp = 0
        # Try robust computation
        for yt, yp in zip(y_true, y_pred):
            if yt == 0 and yp == 0:
                tn += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 1 and yp == 0:
                fn += 1
            elif yt == 1 and yp == 1:
                tp += 1

    # Compute clinical metrics (safe division)
    def safe_div(a, b):
        return float(a) / float(b) if b != 0 else 0.0

    sensitivity = safe_div(tp, tp + fn)  # recall
    specificity = safe_div(tn, tn + fp)
    ppv = prec
    npv = safe_div(tn, tn + fn)

    return {
        "auc": auc,
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def load_teacher_probs(path):
    """
    Load teacher probabilities from a JSONL file where each line has:
    {"mrn": "...", "prob": 0.87, "cot": "..."} or {"mrn": "...", "teacher_prob": 0.87}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    m = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            key = str(j.get("mrn")).strip() if j.get("mrn") is not None else None
            if not key:
                continue
            # Accept either "teacher_prob" or "prob"
            prob = j.get("teacher_prob", j.get("prob", None))
            if prob is None:
                # Skip entries that lack a numeric prob (maybe they only have "cot")
                continue
            try:
                m[key] = float(prob)
            except Exception:
                continue
    return m


# Minimal dataset wrapper for training
class SimpleMTIFDataset(Dataset):
    """
    Minimal dataset holding embeddings, numeric static, categorical static, labels and mrn strings.
    Returns: (emb, num, cat, label, mrn)
    """
    def __init__(self, X, num, cat, y, mrns):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.num = torch.tensor(num, dtype=torch.float32) if num.size else torch.zeros((len(X), 0), dtype=torch.float32)
        self.cat = torch.tensor(cat, dtype=torch.long) if cat.size else torch.zeros((len(X), 0), dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mrns = np.array(mrns).astype(str)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.num[idx], self.cat[idx], self.y[idx], self.mrns[idx]


# -------------------------
# Training function
# -------------------------
def train_with_cot(
    teacher_cot_path="data_processed/mtif_dataset/teacher_cot.jsonl",
    out_model="best_mtif_with_cot.pt",
    epochs=20,
    lr=1e-3,
    distill_weight=0.3,
    ce_weight=0.7,
    batch_size=256,
    val_frac=0.2,
    use_distill=True,
    device_override=None,
    seed=42,
    metrics_csv="train_metrics_with_cot.csv"
):
    """
    Train the MTIF model with optional CoT probability distillation.

    - Splits data into train/val
    - Uses WeightedRandomSampler to balance classes in training
    - For samples with teacher probabilities, adds MSE(probs_student, probs_teacher) * distill_weight
      (skips teacher-loss for rows without teacher prob)
    - Evaluates on validation set each epoch and saves best model by AUC
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=== Loading aligned data ===")
    X, num, cat, y, cat_cardinalities, mrns = load_and_align()
    n = len(y)
    print(f"Loaded dataset rows: {n}")

    # Load teacher probs if requested
    if use_distill:
        print("Loading teacher CoT from:", teacher_cot_path)
        teacher_map = load_teacher_probs(teacher_cot_path)
        # build aligned vector where missing -> np.nan (we'll ignore those in loss)
        teacher_probs = np.array([teacher_map.get(str(m), np.nan) for m in mrns], dtype=np.float32)
        num_teacher_known = int(np.sum(~np.isnan(teacher_probs)))
        print(f"Teacher probabilities found for {num_teacher_known}/{n} rows.")
    else:
        teacher_probs = np.full((n,), np.nan, dtype=np.float32)

    # Split train/val by stratified sampling
    train_idx, val_idx = train_test_split(np.arange(n), test_size=val_frac, stratify=y, random_state=seed)
    print(f"Train rows: {len(train_idx)} | Val rows: {len(val_idx)}")

    # Build datasets
    train_ds = SimpleMTIFDataset(X[train_idx], num[train_idx], cat[train_idx], y[train_idx], [mrns[i] for i in train_idx])
    val_ds = SimpleMTIFDataset(X[val_idx], num[val_idx], cat[val_idx], y[val_idx], [mrns[i] for i in val_idx])

    # Weighted sampler for class imbalance (train only)
    class_counts = np.bincount(y[train_idx].astype(int), minlength=2)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights_per_class = 1.0 / class_counts
    sample_weights = weights_per_class[y[train_idx].astype(int)]
    sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), num_samples=len(train_idx), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = MTIFModel(
        tpse_dim=X.shape[1],
        num_static_num=num.shape[1],
        cat_cardinalities=cat_cardinalities,
        hidden_dim=256,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_with_logits = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss(reduction="mean")

    best_auc = -1.0
    metrics_history = []

    print("\n=== Training ===")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for Xb, numb, catb, yb, mrn_b in train_loader:
            Xb = Xb.to(device)
            numb = numb.to(device)
            catb = catb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits, fused = model(Xb, numb, catb)
            if logits.dim() > 1:
                logits = logits.view(-1)
            probs = torch.sigmoid(logits)

            # supervised loss
            loss_ce = bce_with_logits(logits, yb)

            # distillation loss only for samples that have teacher probs
            loss_kd = torch.tensor(0.0, device=device)
            if use_distill:
                # find teacher probs for this batch using mrn strings
                teacher_vals = []
                mask_indices = []
                for i, m in enumerate(mrn_b):
                    key = str(m)
                    val = teacher_map.get(key, None) if key in (teacher_map if 'teacher_map' in locals() else {}) else None
                    # membership: we precomputed teacher_probs array, but here faster to use mapping
                    # if teacher not present, skip this index
                    if val is not None and not np.isnan(val):
                        teacher_vals.append(val)
                        mask_indices.append(i)
                if len(mask_indices) > 0:
                    teacher_tensor = torch.tensor(np.array(teacher_vals, dtype=np.float32), device=device)
                    pred_subset = probs[mask_indices]
                    # MSE in probability space
                    loss_kd = mse(pred_subset, teacher_tensor)

            # combine losses
            if use_distill:
                loss = ce_weight * loss_ce + distill_weight * loss_kd
            else:
                loss = loss_ce

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        epoch_loss = running_loss / (n_batches if n_batches > 0 else 1)

        # Validation
        model.eval()
        all_probs = []
        all_y = []
        with torch.no_grad():
            for Xv, numv, catv, yv, _ in val_loader:
                Xv = Xv.to(device)
                numv = numv.to(device)
                catv = catv.to(device)
                logits_v, _ = model(Xv, numv, catv)
                if logits_v.dim() > 1:
                    logits_v = logits_v.view(-1)
                probs_v = torch.sigmoid(logits_v).cpu().numpy()
                all_probs.append(probs_v)
                all_y.append(yv.numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        metrics = compute_metrics(all_y, all_probs, threshold=0.5)
        metrics["epoch"] = epoch
        metrics["loss"] = epoch_loss
        metrics_history.append(metrics)

        print(
            f"Epoch {epoch:02d} | loss={epoch_loss:.4f} | AUC={metrics['auc']:.4f} | "
            f"F1={metrics['f1']:.4f} | Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | "
            f"Sens={metrics['sensitivity']:.4f} | Spec={metrics['specificity']:.4f} | "
            f"PPV={metrics['ppv']:.4f} | NPV={metrics['npv']:.4f}"
        )

        # save best model by AUC
        if not np.isnan(metrics["auc"]) and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), out_model)
            print(f"  NEW BEST MODEL SAVED -> {out_model} (AUC={best_auc:.4f})")

    # Save metrics history to CSV
    df_metrics = pd.DataFrame(metrics_history)
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Training complete. Best AUC: {best_auc:.6f}")
    print(f"Metrics written to: {metrics_csv}")

    return df_metrics


# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    # Example: python -m src.train_mtif_with_cot
    # You can edit the flags here or call train_with_cot(...) from another script.
    train_with_cot(
        teacher_cot_path="data_processed/mtif_dataset/teacher_cot.jsonl",
        out_model="best_mtif_with_cot.pt",
        epochs=20,
        lr=1e-3,
        distill_weight=0.3,
        ce_weight=0.7,
        batch_size=512,
        val_frac=0.2,
        use_distill=True,
        device_override=None,
        seed=42,
        metrics_csv="train_metrics_with_cot.csv"
    )
