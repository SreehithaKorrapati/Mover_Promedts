import pandas as pd
from sklearn.model_selection import train_test_split

FILE = "features_final_target.csv"

# Load dataset
df = pd.read_csv(FILE)
print("Dataset shape:", df.shape)

# features and target
X = df.drop(columns=['mrn','post_op_complication'])
y = df['post_op_complication']

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train target distribution:\n", y_train.value_counts(normalize=True))

