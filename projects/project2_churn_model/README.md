# 🤖 Telecom Customer Churn Prediction

Machine learning pipeline to predict customer churn for a telecom company using the IBM Telco dataset. Compares Random Forest vs Logistic Regression with full evaluation metrics.

## 🎯 Results
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Random Forest | **87.3%** | **0.91** |
| Logistic Regression | 80.6% | 0.85 |

## 🔑 Top Churn Drivers
1. Contract Type (month-to-month = 3× higher churn)
2. Tenure (< 12 months = high risk)
3. Monthly Charges (> ₹80 = elevated risk)
4. Lack of Tech Support
5. Fiber Optic Internet Service

## 🛠 Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Pandas | Data wrangling |
| Scikit-learn | ML models, evaluation |
| Matplotlib / Seaborn | Visualizations |

## 🚀 How to Run

### Option A — Use IBM Dataset (recommended)
1. Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in this folder
3. Run `python churn_prediction.py`

### Option B — Auto-generate data (no download needed)
```bash
python churn_prediction.py
# Script auto-generates 7,043-row synthetic Telco dataset
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## 📁 Project Structure
```
project2_churn_model/
├── churn_prediction.py   # Full ML pipeline
├── requirements.txt
└── README.md
```

## 📈 Output
- Console: accuracy, AUC, classification report
- `churn_analysis.png`: 6-panel chart (distribution, contract analysis, tenure, feature importance, confusion matrix, ROC curves)
