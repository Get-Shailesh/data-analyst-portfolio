# 👥 IBM HR Analytics — Attrition Analysis

Deep exploratory data analysis of IBM's HR dataset to uncover why employees leave. Identifies key attrition drivers across departments, job roles, salary bands, and working conditions.

## 📊 Key Findings
- Overall attrition rate: **16.1%**
- Sales Executives have the **highest attrition** (26%+)
- Employees working **overtime** are 2.5× more likely to leave
- Low job satisfaction (score 1–2) correlates with **3× higher** attrition
- Employees earning **< ₹40K/month** have significantly higher churn

## 🛠 Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core |
| Pandas | EDA, groupby analysis |
| Matplotlib | Chart layouts |
| Seaborn | Boxplots, distributions |

## 🚀 How to Run

### Option A — Use IBM Dataset (recommended)
1. Download from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in this folder
3. `python hr_attrition_analysis.py`

### Option B — Auto-generate (no download needed)
```bash
python hr_attrition_analysis.py
# Generates 1,470-row synthetic HR dataset automatically
```

```bash
pip install -r requirements.txt
```

## 📁 Project Structure
```
project4_hr_attrition/
├── hr_attrition_analysis.py   # Full EDA script
├── requirements.txt
└── README.md
```

## 📈 Output Charts
1. Headcount by department & attrition (stacked bar)
2. Attrition rate by job role (horizontal bar)
3. Age distribution by attrition
4. Job satisfaction vs attrition rate
5. Monthly income distribution (boxplot)
6. Overtime impact on attrition
