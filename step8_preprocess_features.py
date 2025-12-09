import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

FILE = "features_final_target.csv"

# Load dataset
df = pd.read_csv(FILE)
X = df.drop(columns=['mrn', 'post_op_complication'])
y = df['post_op_complication']

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Drop datetime-like columns from categorical
datetime_cols = [c for c in categorical_cols if 'datetime' in c.lower() or 'in_or_dttm' in c.lower() or 'out_or_dttm' in c.lower()]
if datetime_cols:
    print("Dropped datetime columns from categorical:", datetime_cols)
    categorical_cols = [c for c in categorical_cols if c not in datetime_cols]

X[numeric_cols] = X[numeric_cols].fillna(0)
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_processed, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

print("Processed feature shape:", X_tensor.shape)
print("Target tensor shape:", y_tensor.shape)

torch.save(X_tensor, r"C:\Users\dell\PycharmProjects\PythonProject1\subset\X_tensor.pt")
torch.save(y_tensor, r"C:\Users\dell\PycharmProjects\PythonProject1\subset\y_tensor.pt")


