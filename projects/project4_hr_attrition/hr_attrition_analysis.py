"""
IBM HR Analytics — Attrition EDA
===================================
Dataset: IBM HR Analytics Employee Attrition & Performance
         Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
         Script auto-generates matching data if file not found.
Tools: Python, Pandas, Matplotlib, Seaborn
Author: Aryan Kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load or Generate Data ─────────────────────────────────────────────────
def generate_ibm_hr(n=1470, seed=42):
    np.random.seed(seed)
    depts = ["Sales","Research & Development","Human Resources"]
    roles = {
        "Sales":                    ["Sales Executive","Sales Representative","Manager"],
        "Research & Development":   ["Research Scientist","Laboratory Technician","Healthcare Representative","Research Director","Manufacturing Director"],
        "Human Resources":          ["Human Resources","Manager"],
    }
    dept_arr  = np.random.choice(depts, n, p=[0.32, 0.64, 0.04])
    role_arr  = [np.random.choice(roles[d]) for d in dept_arr]
    age_arr   = np.random.randint(18, 60, n)
    tenure    = np.random.exponential(7, n).clip(0, 40).astype(int)
    salary    = (np.random.normal(65000, 28000, n)).clip(20000, 200000).astype(int)
    satisfaction = np.random.randint(1, 5, n)
    overtime  = np.random.choice(["Yes","No"], n, p=[0.28, 0.72])
    edu_field = np.random.choice(["Life Sciences","Medical","Marketing",
                                   "Technical Degree","Human Resources","Other"], n)
    work_life = np.random.randint(1, 5, n)
    perf      = np.random.choice([3,4], n, p=[0.85, 0.15])
    distance  = np.random.randint(1, 30, n)
    num_companies = np.random.randint(0, 10, n)

    churn_prob = (
        (overtime == "Yes") * 0.20 +
        (satisfaction <= 2) * 0.20 +
        (tenure < 3) * 0.20 +
        (work_life <= 2) * 0.15 +
        (dept_arr == "Sales") * 0.10 +
        np.random.uniform(0, 0.15, n)
    ).clip(0, 1)
    attrition = np.where(np.random.uniform(size=n) < churn_prob, "Yes", "No")

    return pd.DataFrame({
        "Age": age_arr, "Attrition": attrition,
        "Department": dept_arr, "JobRole": role_arr,
        "MonthlyIncome": salary,
        "JobSatisfaction": satisfaction,
        "YearsAtCompany": tenure,
        "OverTime": overtime,
        "EducationField": edu_field,
        "WorkLifeBalance": work_life,
        "PerformanceRating": perf,
        "DistanceFromHome": distance,
        "NumCompaniesWorked": num_companies,
        "MaritalStatus": np.random.choice(["Single","Married","Divorced"], n, p=[0.32,0.46,0.22]),
        "Gender": np.random.choice(["Male","Female"], n, p=[0.60, 0.40]),
        "EnvironmentSatisfaction": np.random.randint(1,5,n),
        "RelationshipSatisfaction": np.random.randint(1,5,n),
        "StockOptionLevel": np.random.randint(0,4,n),
    })

try:
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    print("✅ Loaded IBM HR dataset from file.")
except FileNotFoundError:
    df = generate_ibm_hr()
    df.to_csv("hr_attrition_data.csv", index=False)
    print("✅ Generated synthetic IBM HR dataset (1,470 rows) → hr_attrition_data.csv")

# ── 2. EDA ────────────────────────────────────────────────────────────────────
print(f"\n📐 Dataset Shape: {df.shape}")
overall_rate = (df["Attrition"]=="Yes").mean() * 100
print(f"🔴 Overall Attrition Rate: {overall_rate:.1f}%")

print("\n📊 Attrition Rate by Department:")
dept_att = df.groupby("Department")["Attrition"].apply(lambda x: (x=="Yes").mean()*100).sort_values(ascending=False)
for dept, rate in dept_att.items():
    print(f"   {dept:<35} {rate:.1f}%")

print("\n📊 Attrition Rate by Job Role:")
role_att = df.groupby("JobRole")["Attrition"].apply(lambda x: (x=="Yes").mean()*100).sort_values(ascending=False)
for role, rate in role_att.items():
    print(f"   {role:<40} {rate:.1f}%")

# ── 3. Visualizations ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=0.95)
fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle("IBM HR Analytics — Attrition Deep Dive", fontsize=16, fontweight="bold")

# 3a. Attrition by Department (stacked bar)
ax = axes[0, 0]
dept_counts = df.groupby(["Department","Attrition"]).size().unstack(fill_value=0)
dept_counts.plot(kind="bar", stacked=True, ax=ax,
                 color=["#4361ee","#ef4444"], width=0.5, edgecolor="white")
ax.set_title("Headcount by Department & Attrition", fontweight="bold")
ax.set_ylabel("Employees")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
ax.legend(["Stayed","Left"])

# 3b. Attrition Rate by Job Role
ax = axes[0, 1]
role_rate = df.groupby("JobRole")["Attrition"].apply(lambda x: (x=="Yes").mean()*100).sort_values()
colors_role = ["#4361ee" if v < 20 else "#ef4444" for v in role_rate.values]
role_rate.plot(kind="barh", ax=ax, color=colors_role, edgecolor="white")
ax.set_title("Attrition Rate by Job Role (%)", fontweight="bold")
ax.set_xlabel("Attrition Rate (%)")
ax.axvline(overall_rate, color="gray", linestyle="--", linewidth=1.5, label=f"Avg: {overall_rate:.1f}%")
ax.legend(fontsize=9)

# 3c. Age distribution by Attrition
ax = axes[1, 0]
df[df["Attrition"]=="No"]["Age"].hist(bins=20, ax=ax, alpha=0.7,
                                       color="#4361ee", label="Stayed", edgecolor="white")
df[df["Attrition"]=="Yes"]["Age"].hist(bins=20, ax=ax, alpha=0.7,
                                        color="#ef4444", label="Left", edgecolor="white")
ax.set_title("Age Distribution by Attrition", fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend()

# 3d. Job Satisfaction vs Attrition
ax = axes[1, 1]
sat_att = df.groupby("JobSatisfaction")["Attrition"].apply(lambda x: (x=="Yes").mean()*100)
bars = ax.bar(["Low (1)","Medium (2)","High (3)","Very High (4)"],
              sat_att.values,
              color=["#ef4444","#f59e0b","#4361ee","#10b981"],
              width=0.5, edgecolor="white")
ax.set_title("Attrition Rate by Job Satisfaction Level", fontweight="bold")
ax.set_ylabel("Attrition Rate (%)")
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontweight="bold")

# 3e. Monthly Income by Attrition (boxplot)
ax = axes[2, 0]
df.boxplot(column="MonthlyIncome", by="Attrition", ax=ax,
           boxprops=dict(color="#4361ee"), medianprops=dict(color="#ef4444",linewidth=2),
           whiskerprops=dict(color="#4361ee"), capprops=dict(color="#4361ee"),
           flierprops=dict(marker="o", color="#4361ee", alpha=0.3, markersize=3))
ax.set_title("Monthly Income Distribution by Attrition", fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Monthly Income (₹)")
plt.sca(ax)
plt.title("Monthly Income Distribution by Attrition", fontweight="bold")

# 3f. Overtime & Work-Life Balance
ax = axes[2, 1]
ot_att = df.groupby("OverTime")["Attrition"].apply(lambda x: (x=="Yes").mean()*100)
bars6 = ax.bar(ot_att.index, ot_att.values,
               color=["#4361ee","#ef4444"], width=0.4, edgecolor="white")
ax.set_title("Attrition Rate: OverTime vs No OverTime", fontweight="bold")
ax.set_ylabel("Attrition Rate (%)")
ax.set_ylim(0, ot_att.max() * 1.3)
for bar in bars6:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig("hr_attrition_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Chart saved: hr_attrition_analysis.png")
