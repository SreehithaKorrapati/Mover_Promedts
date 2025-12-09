import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import numpy as np

# Load dataset 
FILE = "features_final_target.csv"
df = pd.read_csv(FILE)

# Features and target
X = df.drop(columns=['mrn', 'post_op_complication'])
y = df['post_op_complication']
X = X.fillna(0)

# Identify categorical columns 
categorical_cols = [c for c in X.select_dtypes(include=['object']).columns
                    if not any(sub in c for sub in ['in_or_dttm', 'out_or_dttm'])]

drop_cols = [c for c in X.columns if 'or_dttm' in c]
X = X.drop(columns=drop_cols)

#  Train/Validation Split (same as MLP) 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class_counts = np.bincount(y_train)
weights = {0: 1.0 / class_counts[0], 1: 1.0 / class_counts[1]}
sample_weights = y_train.map(weights)

# XGBoost 
X_train_xgb = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_val_xgb = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)

# Ensure same columns
X_val_xgb = X_val_xgb.reindex(columns=X_train_xgb.columns, fill_value=0)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train_xgb, y_train, sample_weight=sample_weights)

y_pred_xgb = xgb_model.predict(X_val_xgb)
y_proba_xgb = xgb_model.predict_proba(X_val_xgb)[:, 1]

print("XGBoost Results")
print(classification_report(y_val, y_pred_xgb, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_xgb))
print("AUROC:", roc_auc_score(y_val, y_proba_xgb))

# LightGBM 
# Convert categorical columns to 'category'
for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

lgb_model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    categorical_feature=categorical_cols
)

y_pred_lgb = lgb_model.predict(X_val)
y_proba_lgb = lgb_model.predict_proba(X_val)[:, 1]

print("\n LightGBM Results ")
print(classification_report(y_val, y_pred_lgb, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_lgb))
print("AUROC:", roc_auc_score(y_val, y_proba_lgb))

