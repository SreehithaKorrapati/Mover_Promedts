import torch
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


X_FILE = "X_tensor.pt"
Y_FILE = "y_tensor.pt"
X_tensor = torch.load(X_FILE)
y_tensor = torch.load(Y_FILE)

# Dataset & DataLoader 
dataset = TensorDataset(X_tensor, y_tensor)
val_loader = DataLoader(dataset, batch_size=64, shuffle=False)

class PostOpMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = PostOpMLP(X_tensor.shape[1])
model.load_state_dict(torch.load("postop_model.pt"))  
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in val_loader:
        preds = model(xb)
        all_preds.append(preds)
        all_labels.append(yb)

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

threshold = 0.5
y_pred = (all_preds >= threshold).astype(int)

print("Classification Report:")
print(classification_report(all_labels, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, y_pred))

# AUROC
try:
    auroc = roc_auc_score(all_labels, all_preds)
    print(f"AUROC: {auroc:.4f}")
except ValueError:
    print("AUROC cannot be computed (maybe only one class present).")


