# === 1. Import Library ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
import joblib  # for saving models

# === 2. Load Dataset ===
df = pd.read_csv("lendingclub_feature engineer.csv")

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
    X[col] = encoder.fit_transform(X[col].astype(str))

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

# === 7. Handling Imbalance with Random Under Sampler (Only on Training Data) ===
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print("Before RUS:", np.bincount(y_train))
print("After RUS :", np.bincount(y_train_res))

# === 8. Define Models ===
models = {
    "LightGBM": LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=-1,
        random_state=42, n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# === 9. Training & Evaluation ===
results = []

for name, model in models.items():
    # Train model
    model.fit(X_train_res, y_train_res)

    # Save model
    model_filename = f"v2/{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"✅ Model '{name}' saved as: {model_filename}")

    # Prediction
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # === Evaluation Metrics ===
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) != 0 else 0
    gmean = np.sqrt(sensitivity * specificity)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "AUC": auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "FP-Rate": fp_rate,
        "G-Mean": gmean
    })

    print(f"\n===== {name} =====")
    print(f"Accuracy   : {acc:.4f}")
    print(f"AUC        : {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"FP-Rate    : {fp_rate:.4f}")
    print(f"G-Mean     : {gmean:.4f}")

# === 10. Save Results ===
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n=== Model Performance Summary ===")
print(results_df)

# === 11. Visualization ===
plt.figure(figsize=(9,5))
plt.bar(results_df["Model"], results_df["Accuracy"], color='skyblue', label="Accuracy")
plt.bar(results_df["Model"], results_df["AUC"], alpha=0.6, color='orange', label="AUC")
plt.xticks(rotation=20)
plt.ylabel("Score")
plt.title("Model Performance Comparison (RUS Applied)")
plt.ylim(0, 1)
plt.legend()
plt.show()

# === 12. Save Training Results to CSV ===
results_df.to_csv("model_results_with_rus.csv", index=False)
print("\n📁 Hasil evaluasi telah disimpan ke: model_results_with_rus.csv")
