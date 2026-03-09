"""
Telecom Customer Churn Prediction
===================================
Dataset: IBM Telco Customer Churn Dataset
         Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
         OR this script auto-generates matching synthetic data so it runs offline.
Tools: Python, Pandas, Scikit-learn, Matplotlib, Seaborn
Author: Aryan Kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load or Generate Dataset ─────────────────────────────────────────────
def generate_telco_data(n=7043, seed=42):
    """Generate synthetic Telco-style data matching IBM dataset structure."""
    np.random.seed(seed)
    contract_types = np.random.choice(["Month-to-month","One year","Two year"],
                                       n, p=[0.55, 0.25, 0.20])
    internet = np.random.choice(["DSL","Fiber optic","No"], n, p=[0.34, 0.44, 0.22])
    tenure = np.random.exponential(scale=32, size=n).clip(1, 72).astype(int)
    monthly = np.random.normal(65, 30, n).clip(18, 120).round(2)
    churn_prob = (
        (contract_types == "Month-to-month") * 0.35 +
        (internet == "Fiber optic") * 0.15 +
        (tenure < 12) * 0.25 +
        (monthly > 80) * 0.15 +
        np.random.uniform(0, 0.1, n)
    ).clip(0, 1)
    churn = (np.random.uniform(size=n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "gender": np.random.choice(["Male","Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": np.random.choice(["Yes","No"], n),
        "Dependents": np.random.choice(["Yes","No"], n, p=[0.30, 0.70]),
        "tenure": tenure,
        "PhoneService": np.random.choice(["Yes","No"], n, p=[0.90, 0.10]),
        "MultipleLines": np.random.choice(["Yes","No","No phone service"], n, p=[0.42, 0.48, 0.10]),
        "InternetService": internet,
        "OnlineSecurity": np.random.choice(["Yes","No","No internet service"], n, p=[0.29, 0.49, 0.22]),
        "TechSupport": np.random.choice(["Yes","No","No internet service"], n, p=[0.29, 0.49, 0.22]),
        "Contract": contract_types,
        "PaperlessBilling": np.random.choice(["Yes","No"], n, p=[0.59, 0.41]),
        "PaymentMethod": np.random.choice(
            ["Electronic check","Mailed check","Bank transfer","Credit card"], n,
            p=[0.34, 0.23, 0.22, 0.21]
        ),
        "MonthlyCharges": monthly,
        "TotalCharges": (monthly * tenure + np.random.normal(0, 50, n)).clip(0).round(2),
        "Churn": pd.Series(churn).map({1: "Yes", 0: "No"}),
    })
    return df

try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("✅ Loaded IBM Telco dataset from file.")
except FileNotFoundError:
    df = generate_telco_data()
    df.to_csv("telco_churn_data.csv", index=False)
    print("✅ Generated synthetic Telco dataset (7,043 rows) → telco_churn_data.csv")

# ── 2. EDA ───────────────────────────────────────────────────────────────────
print(f"\n📐 Dataset Shape: {df.shape}")
print(f"🔴 Churn Rate: {(df['Churn']=='Yes').mean()*100:.1f}%")
print(f"\n🔍 Missing Values:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
df_model = df.copy()
df_model["TotalCharges"] = pd.to_numeric(df_model["TotalCharges"], errors="coerce").fillna(0)
df_model.drop("customerID", axis=1, inplace=True)
df_model["Churn_Binary"] = (df_model["Churn"] == "Yes").astype(int)
df_model.drop("Churn", axis=1, inplace=True)

cat_cols = df_model.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop("Churn_Binary", axis=1)
y = df_model["Churn_Binary"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                      random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4. Train Models ──────────────────────────────────────────────────────────
print("\n🤖 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_acc   = accuracy_score(y_test, rf_preds)
rf_auc   = roc_auc_score(y_test, rf_proba)
print(f"   ✅ Accuracy: {rf_acc*100:.1f}%  |  AUC-ROC: {rf_auc:.3f}")

print("\n🤖 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_preds = lr.predict(X_test_s)
lr_proba = lr.predict_proba(X_test_s)[:, 1]
lr_acc   = accuracy_score(y_test, lr_preds)
lr_auc   = roc_auc_score(y_test, lr_proba)
print(f"   ✅ Accuracy: {lr_acc*100:.1f}%  |  AUC-ROC: {lr_auc:.3f}")

print("\n📋 Random Forest Classification Report:")
print(classification_report(y_test, rf_preds, target_names=["Not Churned","Churned"]))

# ── 5. Visualizations ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Telecom Customer Churn Analysis", fontsize=16, fontweight="bold")

# 5a. Churn distribution
ax = axes[0, 0]
churn_counts = df["Churn"].value_counts()
bars = ax.bar(churn_counts.index, churn_counts.values,
              color=["#4361ee","#ef4444"], width=0.4, edgecolor="white")
ax.set_title("Churn Distribution", fontweight="bold")
ax.set_ylabel("Customers")
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
            f"{bar.get_height():,}", ha="center", va="bottom", fontsize=11)

# 5b. Churn by Contract Type
ax = axes[0, 1]
ct = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack() * 100
ct["Yes"].plot(kind="bar", ax=ax, color="#4361ee", width=0.5, edgecolor="white")
ax.set_title("Churn Rate by Contract Type", fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
for p in ax.patches:
    ax.text(p.get_x()+p.get_width()/2, p.get_height()+0.3,
            f"{p.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

# 5c. Tenure distribution by churn
ax = axes[0, 2]
churned = df[df["Churn"]=="Yes"]["tenure"]
stayed  = df[df["Churn"]=="No"]["tenure"]
ax.hist(stayed,  bins=30, alpha=0.6, color="#4361ee", label="Stayed",  edgecolor="white")
ax.hist(churned, bins=30, alpha=0.7, color="#ef4444", label="Churned", edgecolor="white")
ax.set_title("Tenure Distribution by Churn", fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Count")
ax.legend()

# 5d. Feature Importance
ax = axes[1, 0]
feat_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True).tail(10)
colors_fi = ["#4361ee" if v >= feat_imp.values[-3] else "#a5b4fc" for v in feat_imp.values]
feat_imp.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
ax.set_title("Top 10 Feature Importances (Random Forest)", fontweight="bold")
ax.set_xlabel("Importance Score")

# 5e. Confusion Matrix
ax = axes[1, 1]
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Churned","Churned"],
            yticklabels=["Not Churned","Churned"], ax=ax, linewidths=0.5)
ax.set_title(f"Confusion Matrix (RF — Acc: {rf_acc*100:.1f}%)", fontweight="bold")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")

# 5f. ROC Curves
ax = axes[1, 2]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
ax.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC={rf_auc:.3f})", color="#4361ee", lw=2)
ax.plot(lr_fpr, lr_tpr, label=f"Logistic Reg (AUC={lr_auc:.3f})", color="#7209b7", lw=2, linestyle="--")
ax.plot([0,1],[0,1], color="gray", linestyle=":", lw=1.5)
ax.set_title("ROC Curve Comparison", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.fill_between(rf_fpr, rf_tpr, alpha=0.05, color="#4361ee")

plt.tight_layout()
plt.savefig("churn_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Chart saved: churn_analysis.png")
