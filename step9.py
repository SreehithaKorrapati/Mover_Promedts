import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, random_split

# Paths to saved tensors
X_FILE = r"C:\Users\dell\PycharmProjects\PythonProject1\subset\X_tensor.pt"
Y_FILE = r"C:\Users\dell\PycharmProjects\PythonProject1\subset\y_tensor.pt"

# Load tensors
X_tensor = torch.load(X_FILE)
y_tensor = torch.load(Y_FILE)

# Ensure y is float32
y_tensor = y_tensor.float()

# Dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Train/Validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# Handle class imbalance on the training set
train_targets = torch.cat([y for _, y in train_dataset], dim=0)
class_counts = torch.bincount(train_targets.flatten().long())
weights = 1.0 / class_counts.float()
sample_weights = weights[train_targets.flatten().long()]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define a simple MLP model
class PostOpMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = PostOpMLP(X_tensor.shape[1])

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
# Save the trained model
MODEL_FILE = r"C:\Users\dell\PycharmProjects\PythonProject1\subset\postop_model.pt"
torch.save(model.state_dict(), MODEL_FILE)
print(f"Model saved to: {MODEL_FILE}")
