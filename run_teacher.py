# src/run_teacher.py
import json
import torch
import numpy as np
import pandas as pd
import os
import sys

from src.mtif_model import MTIFModel
from src.train_mtif_baseline import load_and_align


def run_teacher():
    print("\n Loading aligned MTIF dataset")
    try:
        # call without keyword args — unpack the 6 returned items
        X, num, cat, y, cat_cardinalities, mrn_list = load_and_align()
    except TypeError as e:
        print("TypeError calling load_and_align():", e)
        print("This means load_and_align() signature does not match expected. Exiting.")
        return
    except Exception as e:
        print("Error loading aligned data:", e)
        return

    print(f"  Embeddings: {X.shape}")
    print(f"  Numeric:    {num.shape}")
    print(f"  Categorical:{cat.shape}")
    print(f"  Labels:     {y.shape}")
    print(f"  MRNs:       {len(mrn_list)}")
    print(f"  Cat card:   {cat_cardinalities}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    model = MTIFModel(
        tpse_dim=X.shape[1],
        num_static_num=num.shape[1],
        cat_cardinalities=cat_cardinalities,
        hidden_dim=256,
        dropout=0.1,
    ).to(device)

    checkpoint_path = "best_mtif.pt"
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Teacher checkpoint not found at: {checkpoint_path}")
        print("saved the model (train runs should create it).")
        return

    print("\n Loading teacher model")
    try:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print("Teacher checkpoint loaded successfully.")
    except Exception as e:
        print("ERROR loading teacher model:", e)
        return

    model.eval()

    # Convert to tensors 
    print("\n Predicting teacher probabilities")
    try:
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        num_t = torch.tensor(num, dtype=torch.float32).to(device)
        cat_t = torch.tensor(cat, dtype=torch.long).to(device)

        with torch.no_grad():
            logits, _ = model(X_t, num_t, cat_t)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    except Exception as e:
        print("ERROR during teacher inference:", e)
        return

    print("Teacher prediction complete. Number of predictions:", len(probs))

    # Write JSONL 
    out_path = os.path.join("data_processed", "mtif_dataset", "teacher_cot.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"\n=== Step 4: Writing teacher CoT file to: {out_path} ===")
    try:
        with open(out_path, "w", encoding="utf8") as f:
            for mrn, p in zip(mrn_list, probs):
                j = {"mrn": str(mrn), "teacher_prob": float(p)}
                f.write(json.dumps(j) + "\n")
        print("Teacher CoT saved.")
    except Exception as e:
        print("ERROR writing teacher CoT file:", e)
        return

    print("\n successful.\n")


if __name__ == "__main__":
    run_teacher()

