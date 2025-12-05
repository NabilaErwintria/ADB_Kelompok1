# === 1. Import Library ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# === 2. Load Dataset ===
df = pd.read_csv("lendingclub_feature engineer_tanpa_outlier.csv")

# === 3. Feature & Target ===
y = df["loan_condition"]  # target: good/bad

X = df[[
    "loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate",
    "installment", "grade", "home_ownership", "annual_inc",
    "verification_status", "dti",
    "inc_category", "emp_length_int", "int_payments"
]]

# === 4. Encode Categorical Columns ===
categorical_cols = [
    "term", "grade", "home_ownership", "verification_status",
    "inc_category", "int_payments"
]

encoder = LabelEncoder()
for col in categorical_cols:
    X.loc[:, col] = encoder.fit_transform(X[col].astype(str))

# Encode target
y = LabelEncoder().fit_transform(y)

# === 5. Split Train & Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 6. Scaling ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 7. Train Random Forest Only (No SMOTE, No RUS) ===
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# === 8. Save Model ===
joblib.dump(model, "random_forest_model.pkl")
print("✅ Random Forest model saved as: random_forest_model.pkl")

# === 9. Prediction ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# === 10. Confusion Matrix ===
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# === 11. Evaluation Metrics ===
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
fp_rate = fp / (fp + tn) if (fp + tn) != 0 else 0
gmean = np.sqrt(sensitivity * specificity)

print("\n===== Random Forest Evaluation =====")
print(f"Accuracy   : {acc:.4f}")
print(f"AUC        : {auc:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"FP-Rate    : {fp_rate:.4f}")
print(f"G-Mean     : {gmean:.4f}")

# === 12. Visualization ===
plt.figure(figsize=(5, 4))
plt.bar(["Accuracy", "AUC"], [acc, auc], color=['skyblue', 'orange'])
plt.title("Random Forest Performance (No SMOTE / RUS)")
plt.ylim(0, 1)
plt.show()

# === 13. Save Results to CSV ===
results_df = pd.DataFrame([{
    "Model": "Random Forest",
    "Accuracy": acc,
    "AUC": auc,
    "Sensitivity": sensitivity,
    "Specificity": specificity,
    "FP-Rate": fp_rate,
    "G-Mean": gmean
}])

results_df.to_csv("rf_results_no_smote.csv", index=False)
print("\n📁 Hasil evaluasi telah disimpan ke: rf_results_no_smote.csv")
