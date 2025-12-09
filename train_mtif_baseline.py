import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from src.mtif_model import MTIFModel


# Dataset Wrapper
class MTIFDataset(Dataset):
    """
    PyTorch dataset wrapper for MTIF fused model:
        tpse_emb: Time-series pooled embedding
        numeric_static: Numeric static features
        cat_static: Encoded categorical static features
        label: Binary label (float)
        mrn: Patient MRN string
    """
    def __init__(self, X, y, num_static, cat_static, mrns):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.num_static = torch.tensor(num_static, dtype=torch.float32)

        if cat_static.size == 0:
            self.cat_static = torch.zeros((self.X.size(0), 0), dtype=torch.long)
        else:
            self.cat_static = torch.tensor(cat_static, dtype=torch.long)

        self.mrns = np.array(mrns).astype(str)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.num_static[idx],
            self.cat_static[idx],
            self.y[idx],
            self.mrns[idx],
        )


# Focal Loss 
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def _read_csv_flex(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


# Build CoT teacher embeddings using TF-IDF + PCA
def build_cot_embeddings(cot_path, mrn_list, out_dim):
    if not SKLEARN_AVAILABLE:
        print("  sklearn not available — skipping CoT distillation.")
        return {}

    if not os.path.exists(cot_path):
        print("  No teacher CoT file found at:", cot_path)
        return {}

    texts, mrns = [], []
    with open(cot_path, "r", encoding="utf8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except:
                continue
            if "mrn" in j and "cot" in j:
                mrns.append(str(j["mrn"]))
                texts.append(j["cot"])

    if len(texts) == 0:
        print("  Teacher CoT file empty — skipping.")
        return {}

    tf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    Xtf = tf.fit_transform(texts).toarray()

    pca = PCA(n_components=min(out_dim, Xtf.shape[1], Xtf.shape[0]))
    Xproj = pca.fit_transform(Xtf)

    # padding
    if Xproj.shape[1] < out_dim:
        pad = np.zeros((Xproj.shape[0], out_dim - Xproj.shape[1]), dtype=np.float32)
        Xproj = np.concatenate([Xproj.astype(np.float32), pad], axis=1)
    else:
        Xproj = Xproj[:, :out_dim].astype(np.float32)

    cot_map = {str(m): Xproj[i] for i, m in enumerate(mrns)}

    found = sum(1 for m in mrn_list if m in cot_map)
    print(f"  Built CoT embeddings: {len(cot_map)} items | {found} matched MRNs")

    return cot_map


# Load and align data across embeddings, labels, static features
def load_and_align():
    base = "data_processed/"

    # Load embeddings
    print("Loading embeddings...")
    emb_path = os.path.join(base, "embeddings", "pooled_embeddings.npy")
    mrns_path = os.path.join(base, "embeddings", "mrns.csv")
    if not os.path.exists(emb_path) or not os.path.exists(mrns_path):
        raise FileNotFoundError("pooled_embeddings.npy or mrns.csv missing")

    emb = np.load(emb_path)
    emb_mrn = _read_csv_flex(mrns_path)

    mrn_col = None
    for c in emb_mrn.columns:
        if c.lower() == "mrn":
            mrn_col = c
            break
    if mrn_col is None:
        raise RuntimeError("No MRN column in mrns.csv")

    emb_mrn = emb_mrn.rename(columns={mrn_col: "mrn"})
    emb_mrn["mrn"] = emb_mrn["mrn"].astype(str).str.strip()
    df_emb = pd.DataFrame({"mrn": emb_mrn["mrn"], "idx": np.arange(len(emb_mrn))})

    # Load labels
    print("Loading labels...")
    cand_labels = [
        os.path.join(base, "mtif_dataset", "labels.csv"),
        os.path.join(base, "final", "labels.csv"),
        os.path.join(base, "intermediate", "labels.csv"),
    ]
    labels = None
    for p in cand_labels:
        if os.path.exists(p):
            labels = _read_csv_flex(p)
            print(f"  Using labels: {p}")
            break
    if labels is None:
        raise FileNotFoundError("No labels CSV found")

    mrn_col = None
    for c in labels.columns:
        if c.lower() == "mrn":
            mrn_col = c
            break
    if mrn_col is None:
        raise RuntimeError("MRN column missing in labels")

    labels = labels.rename(columns={mrn_col: "mrn"})
    labels["mrn"] = labels["mrn"].astype(str).str.strip()

    if "complication_flag" in labels.columns:
        label_col = "complication_flag"
    elif "label" in labels.columns:
        label_col = "label"
    elif "complication" in labels.columns:
        label_col = "complication"
    else:
        other_cols = [c for c in labels.columns if c != "mrn"]
        label_col = other_cols[0]
        print(f"  Warning: using fallback column {label_col}")

    labels[label_col] = pd.to_numeric(labels[label_col], errors="coerce").fillna(0).astype(int)

    # Numeric static
    print("Loading numeric static...")
    num_candidates = [
        os.path.join(base, "mtif_dataset", "numeric_static_z.csv"),
        os.path.join(base, "mtif_dataset", "numeric_static.csv"),
        os.path.join(base, "final", "numeric_static.csv"),
    ]
    num = None
    for p in num_candidates:
        if os.path.exists(p):
            num = _read_csv_flex(p)
            print(f"  Using numeric static: {p}")
            break
    if num is None:
        print("  No numeric static found, using empty")
        num = pd.DataFrame({"mrn": df_emb["mrn"]})

    if "mrn" not in num.columns:
        for c in num.columns:
            if c.lower() == "mrn":
                num = num.rename(columns={c: "mrn"})
                break
    if "mrn" not in num.columns:
        if num.shape[0] == emb.shape[0]:
            num["mrn"] = df_emb["mrn"].values
            print("  Warning: numeric static had no MRN column")
        else:
            num = pd.DataFrame({"mrn": df_emb["mrn"]})
    num["mrn"] = num["mrn"].astype(str).str.strip()

    # Categorical static
    print("Loading categorical static...")
    cat_enc_path = os.path.join(base, "mtif_dataset", "categorical_static_encoded.csv")
    cat_raw_path = os.path.join(base, "mtif_dataset", "categorical_static.csv")

    if os.path.exists(cat_enc_path):
        cat = _read_csv_flex(cat_enc_path)
        print(f"  Using encoded categorical static: {cat_enc_path}")
    elif os.path.exists(cat_raw_path):
        cat = _read_csv_flex(cat_raw_path)
        print(f"  Using raw categorical static: {cat_raw_path}")
    else:
        print("  No categorical static found, using empty")
        cat = pd.DataFrame({"mrn": df_emb["mrn"]})

    if "mrn" not in cat.columns:
        for c in cat.columns:
            if c.lower() == "mrn":
                cat = cat.rename(columns={c: "mrn"})
                break
    if "mrn" not in cat.columns:
        if cat.shape[0] == emb.shape[0]:
            cat["mrn"] = df_emb["mrn"].values
            print("  Warning: categorical static had no MRN column")
        else:
            cat = pd.DataFrame({"mrn": df_emb["mrn"]})
    cat["mrn"] = cat["mrn"].astype(str).str.strip()

    # Encode non-numeric categorical columns
    for col in cat.columns:
        if col == "mrn":
            continue
        if pd.api.types.is_numeric_dtype(cat[col]):
            cat[col] = pd.to_numeric(cat[col], errors="coerce").fillna(0).astype(int)
        else:
            cat[col], _ = pd.factorize(cat[col].astype(str).fillna("NA"))
            cat[col] = cat[col].astype(int)

    # Encoders metadata
    enc_path = os.path.join(base, "mtif_dataset", "encoders.json")
    if os.path.exists(enc_path):
        with open(enc_path, "r") as f:
            enc = json.load(f)
        encoders = enc.get("encoders", {})
        cat_cardinalities = [v.get("vocab_size", None) for v in encoders.values()]
    else:
        encoders = {}
        cat_cardinalities = []

    # Merge all by MRN
    print("Aligning datasets...")
    df = labels[["mrn", label_col]].merge(num, on="mrn", how="inner")
    df = df.merge(cat, on="mrn", how="inner")
    df = df.merge(df_emb, on="mrn", how="inner")

    df = df.sort_values("idx").reset_index(drop=True)

    X_emb = emb[df["idx"].values]
    y = df[label_col].values.astype(float)
    mrn_list = df["mrn"].astype(str).tolist()

    num_cols = [c for c in num.columns if c != "mrn" and c in df.columns]
    if len(num_cols) == 0:
        num_arr = np.zeros((len(df), 0), dtype=np.float32)
    else:
        num_arr = df[num_cols].astype(float).values

    cat_cols = [c for c in cat.columns if c != "mrn" and c in df.columns]
    if len(cat_cols) == 0:
        cat_arr = np.zeros((len(df), 0), dtype=np.int64)
    else:
        cat_arr = df[cat_cols].fillna(0).astype(int).values

    # Build cardinalities
    if len(cat_cardinalities) < len(cat_cols):
        inferred = []
        for col in cat_cols:
            inferred.append(int(df[col].max()) + 1)
        def get_card(i):
            if i < len(cat_cardinalities) and cat_cardinalities[i] is not None:
                return int(cat_cardinalities[i])
            return int(inferred[i])
        cat_cardinalities = [get_card(i) for i in range(len(cat_cols))]

    print("Final shapes:")
    print("  Embeddings:", X_emb.shape)
    print("  Numeric:", num_arr.shape)
    print("  Categorical:", cat_arr.shape)
    print("  Labels:", y.shape)
    print("  Cat cardinalities:", cat_cardinalities)

    return X_emb, num_arr, cat_arr, y, cat_cardinalities, mrn_list


# Training pipeline
def train():
    X, num, cat, y, cat_cardinalities, mrn_list = load_and_align()

    # CoT distillation
    DISTILL_WEIGHT = 0.1
    COT_PATH = "data_processed/mtif_dataset/teacher_cot.jsonl"
    cot_map = build_cot_embeddings(COT_PATH, mrn_list, out_dim=256) if SKLEARN_AVAILABLE else {}

    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )

    train_dataset = MTIFDataset(
        X[train_idx], y[train_idx], num[train_idx], cat[train_idx],
        [mrn_list[i] for i in train_idx]
    )
    val_dataset = MTIFDataset(
        X[val_idx], y[val_idx], num[val_idx], cat[val_idx],
        [mrn_list[i] for i in val_idx]
    )

    class_counts = np.bincount(y.astype(int), minlength=2)
    class_counts = np.where(class_counts == 0, 1
    , class_counts)
    weight_per_class = 1.0 / class_counts
    sample_weights = weight_per_class[y.astype(int)]

    train_weights = torch.as_tensor(sample_weights[train_idx], dtype=torch.double)
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_idx), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Model
    model = MTIFModel(
        tpse_dim=X.shape[1],
        num_static_num=num.shape[1],
        cat_cardinalities=cat_cardinalities,
        hidden_dim=256,
        dropout=0.1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = FocalLoss(alpha=0.75)
    mse_loss = nn.MSELoss()

    best_auc = 0.0
    epochs = 20

    print("\nTraining...\n")
    print("  Distillation active:", bool(cot_map))

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            Xb, num_b, cat_b, yb, mrn_b = batch
            Xb, num_b, cat_b, yb = (
                Xb.to(device),
                num_b.to(device),
                cat_b.to(device),
                yb.to(device),
            )

            logits, fused = model(Xb, num_b, cat_b)
            loss = loss_fn(logits, yb)

            # distillation
            distill_term = 0.0
            if cot_map:
                target_list = []
                mask_idx = []
                for i, mrn in enumerate(mrn_b):
                    key = str(mrn.item()) if isinstance(mrn, torch.Tensor) else str(mrn)
                    if key in cot_map:
                        target_list.append(cot_map[key])
                        mask_idx.append(i)
                if len(target_list) > 0:
                    target_tensor = torch.tensor(
                        np.stack(target_list, axis=0),
                        dtype=torch.float32,
                        device=device,
                    )
                    pred_subset = fused[mask_idx, :]
                    distill_term = mse_loss(pred_subset, target_tensor) * DISTILL_WEIGHT
                    loss = loss + distill_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Validation
        model.eval()
        preds = []
        true = []

        with torch.no_grad():
            for batch in val_loader:
                Xb, num_b, cat_b, yb, mrn_b = batch
                logits, _ = model(Xb.to(device), num_b.to(device), cat_b.to(device))
                prob = torch.sigmoid(logits).cpu().numpy()
                preds.extend(prob.tolist())
                true.extend(yb.numpy().tolist())

        try:
            auc = roc_auc_score(true, preds)
        except:
            auc = float("nan")

        # Additional medical metrics
        y_pred_bin = (np.array(preds) >= 0.5).astype(int)

        precision = precision_score(true, y_pred_bin, zero_division=0)
        recall = recall_score(true, y_pred_bin, zero_division=0)
        f1 = f1_score(true, y_pred_bin, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(true, y_pred_bin).ravel()

        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        ppv = tp / (tp + fp + 1e-9)
        npv = tn / (tn + fn + 1e-9)

        print(
            f"Epoch {epoch:02d} | loss={np.mean(losses):.4f} | "
            f"AUC={auc:.4f} | F1={f1:.4f} | Precision={precision:.4f} | "
            f"Recall={recall:.4f} | Sens={sensitivity:.4f} | Spec={specificity:.4f} | "
            f"PPV={ppv:.4f} | NPV={npv:.4f}"
        )

        if auc > best_auc and not np.isnan(auc):
            best_auc = auc
            torch.save(model.state_dict(), "best_mtif.pt")
            print(f"  NEW BEST MODEL SAVED (AUC = {best_auc:.4f})")

    print("\nTraining complete.")
    print(f"Best AUC = {best_auc:.4f}")


if __name__ == "__main__":
    train()

