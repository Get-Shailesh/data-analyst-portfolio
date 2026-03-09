# 📣 Marketing Funnel & ROI Analysis

Full-funnel marketing analytics across 6 acquisition channels. Computes ROAS, CPA, CTR, and monthly revenue trends. Helps identify highest-performing channels for budget reallocation.

## 📊 Key Findings
| Channel | ROAS | Avg ROI | Best For |
|---------|------|---------|----------|
| Email | 5.2x | 380% | Conversion |
| Direct | 4.8x | 340% | High-value customers |
| Referral | 4.1x | 290% | Low CPA |
| SEO | 3.2x | 180% | Volume |
| Paid Ads | 2.1x | 95% | Scale |
| Social Media | 1.8x | 65% | Awareness |

**Recommendation:** Shift 30% of Paid Ads budget to Email + Referral to improve blended ROAS by ~0.8x.

## 🛠 Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core |
| Pandas | Aggregations, pivot tables |
| Matplotlib | Charts |
| Seaborn | Styling |

## 🚀 How to Run

```bash
pip install -r requirements.txt
python marketing_funnel_analysis.py
```

Dataset is auto-generated — no external file needed.

## 📁 Project Structure
```
project6_marketing_funnel/
├── marketing_funnel_analysis.py   # Full analysis
├── requirements.txt
└── README.md
```

## 📈 Output Charts
1. Total annual revenue by channel
2. ROAS by channel (with break-even line)
3. Monthly revenue trend (all channels)
4. Overall marketing funnel (impressions → customers)
5. Cost Per Acquisition (CPA) by channel
6. Spend vs revenue scatter (all channels)
