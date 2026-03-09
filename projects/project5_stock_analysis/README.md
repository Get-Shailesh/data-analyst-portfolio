# 📈 FAANG Stock Price Volatility Analysis

Quantitative analysis of FAANG stocks (Apple, Google, Microsoft, Meta, Amazon) throughout 2023. Computes annualized volatility, Sharpe ratios, max drawdowns, and rolling risk metrics using real Yahoo Finance data.

## 📊 Key Findings (2023)
| Stock | Total Return | Ann. Volatility | Sharpe |
|-------|-------------|----------------|--------|
| META  | +194%       | 38.2%           | 2.91   |
| MSFT  | +57%        | 22.4%           | 1.84   |
| GOOGL | +58%        | 25.1%           | 1.67   |
| AMZN  | +81%        | 28.6%           | 1.72   |
| AAPL  | +48%        | 19.8%           | 1.62   |

## 🛠 Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core |
| yfinance | Real-time Yahoo Finance data |
| Pandas | Time-series manipulation |
| Matplotlib | Charting |

## 🚀 How to Run

```bash
pip install -r requirements.txt
python stock_volatility_analysis.py
```

**Online mode:** Downloads live data from Yahoo Finance via `yfinance`  
**Offline mode:** Uses embedded 2023 FAANG trajectories automatically

## 📁 Project Structure
```
project5_stock_analysis/
├── stock_volatility_analysis.py   # Full analysis
├── requirements.txt
└── README.md
```

## 📈 Output Charts
1. Normalized price performance (base 100)
2. Daily returns distribution (violin plot)
3. Annualized volatility vs Sharpe ratio
4. 30-day rolling volatility over time
