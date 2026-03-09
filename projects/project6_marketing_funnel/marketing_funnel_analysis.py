"""
Marketing Channel Funnel & ROI Analysis
==========================================
Dataset: Synthetically generated marketing spend data (industry-realistic)
Tools: Python, Pandas, Matplotlib, Seaborn
Author: Aryan Kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. Generate Dataset ───────────────────────────────────────────────────────
np.random.seed(42)

channels = ["Social Media","Search (SEO)","Email","Paid Ads","Referral","Direct"]
months   = pd.date_range("2023-01-01", periods=12, freq="MS")

# Monthly spend per channel
spend_base = {
    "Social Media": 12000, "Search (SEO)": 8000, "Email": 3500,
    "Paid Ads": 18000,     "Referral": 2000,     "Direct": 500,
}

# Conversion rates (impressions → clicks → leads → customers)
conv_rates = {
    "Social Media": {"ctr":0.090, "l2c":0.025, "c2s":0.18},
    "Search (SEO)": {"ctr":0.160, "l2c":0.040, "c2s":0.22},
    "Email":        {"ctr":0.240, "l2c":0.080, "c2s":0.35},
    "Paid Ads":     {"ctr":0.150, "l2c":0.030, "c2s":0.20},
    "Referral":     {"ctr":0.200, "l2c":0.060, "c2s":0.30},
    "Direct":       {"ctr":0.300, "l2c":0.100, "c2s":0.40},
}

avg_order_value = {
    "Social Media":1200,"Search (SEO)":1800,"Email":2100,
    "Paid Ads":1500,"Referral":2400,"Direct":2800,
}

rows = []
for month in months:
    seasonal = 1 + 0.15 * np.sin(2 * np.pi * month.month / 12)
    for ch in channels:
        spend = spend_base[ch] * seasonal * np.random.uniform(0.9, 1.1)
        impressions = int(spend * np.random.uniform(40, 80))
        ctr = conv_rates[ch]["ctr"] * np.random.uniform(0.85, 1.15)
        clicks = int(impressions * ctr)
        leads  = int(clicks * conv_rates[ch]["l2c"] * np.random.uniform(0.9, 1.1))
        customers = int(leads * conv_rates[ch]["c2s"] * np.random.uniform(0.9, 1.1))
        revenue = customers * avg_order_value[ch] * np.random.uniform(0.95, 1.05)
        rows.append({
            "Month": month, "Channel": ch,
            "Spend": round(spend, 2),
            "Impressions": impressions, "Clicks": clicks,
            "Leads": leads, "Customers": customers,
            "Revenue": round(revenue, 2),
        })

df = pd.DataFrame(rows)
df["ROI"] = ((df["Revenue"] - df["Spend"]) / df["Spend"]).round(3)
df["CPA"]  = (df["Spend"] / df["Customers"].replace(0, np.nan)).round(2)
df["CPL"]  = (df["Spend"] / df["Leads"].replace(0, np.nan)).round(2)
df["CTR"]  = (df["Clicks"] / df["Impressions"] * 100).round(2)

df.to_csv("marketing_funnel_data.csv", index=False)
print("✅ Dataset created: marketing_funnel_data.csv")
print(f"   Shape: {df.shape}")

# ── 2. Summary Stats ───────────────────────────────────────────────────────────
agg = df.groupby("Channel").agg(
    Total_Spend=("Spend","sum"),
    Total_Revenue=("Revenue","sum"),
    Total_Customers=("Customers","sum"),
    Avg_ROI=("ROI","mean"),
    Avg_CPA=("CPA","mean"),
    Avg_CTR=("CTR","mean"),
).round(2)
agg["ROAS"] = (agg["Total_Revenue"] / agg["Total_Spend"]).round(2)

print("\n📊 Channel Performance Summary:")
print(agg[["Total_Spend","Total_Revenue","ROAS","Avg_ROI","Avg_CPA"]].to_string())

best_roi = agg["Avg_ROI"].idxmax()
best_vol = agg["Total_Revenue"].idxmax()
print(f"\n🏆 Best ROI Channel: {best_roi} ({agg.loc[best_roi,'Avg_ROI']*100:.0f}% avg ROI)")
print(f"💰 Highest Revenue Channel: {best_vol} (₹{agg.loc[best_vol,'Total_Revenue']:,.0f})")

# ── 3. Visualizations ──────────────────────────────────────────────────────────
palette = ["#4361ee","#7209b7","#10b981","#ef4444","#f59e0b","#06b6d4"]
ch_colors = dict(zip(channels, palette))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Marketing Funnel & ROI Analysis — 2023", fontsize=16, fontweight="bold")

# 3a. Total Revenue by Channel
ax = axes[0, 0]
rev = agg["Total_Revenue"].sort_values(ascending=True)
colors_bar = [ch_colors[c] for c in rev.index]
bars = ax.barh(rev.index, rev.values / 1000, color=colors_bar, edgecolor="white", height=0.6)
ax.set_title("Total Annual Revenue by Channel", fontweight="bold")
ax.set_xlabel("Revenue (₹ thousands)")
for bar in bars:
    ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
            f"₹{bar.get_width():.0f}k", va="center", fontsize=9)

# 3b. ROAS by Channel
ax = axes[0, 1]
roas = agg["ROAS"].sort_values(ascending=False)
bar_colors = ["#10b981" if v >= 3 else "#4361ee" if v >= 2 else "#f59e0b" for v in roas.values]
bars2 = ax.bar(roas.index, roas.values, color=bar_colors, width=0.55, edgecolor="white")
ax.axhline(1, color="red", linestyle="--", linewidth=1, label="Break-even (ROAS=1)")
ax.set_title("Return on Ad Spend (ROAS)", fontweight="bold")
ax.set_ylabel("ROAS (₹ revenue per ₹1 spend)")
ax.set_xticklabels(roas.index, rotation=20, ha="right")
ax.legend(fontsize=9)
for bar in bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f"{bar.get_height():.1f}x", ha="center", fontsize=9, fontweight="bold")

# 3c. Monthly Revenue Trend by Channel
ax = axes[0, 2]
monthly_rev = df.pivot_table(index="Month", columns="Channel", values="Revenue", aggfunc="sum")
for ch in channels:
    ax.plot(monthly_rev.index, monthly_rev[ch]/1000, label=ch,
            color=ch_colors[ch], linewidth=2, marker="o", markersize=3)
ax.set_title("Monthly Revenue Trend by Channel", fontweight="bold")
ax.set_ylabel("Revenue (₹ thousands)")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
ax.legend(fontsize=8, ncol=2)
ax.tick_params(axis="x", rotation=45)

# 3d. Funnel (total across all channels)
ax = axes[1, 0]
funnel_data = {
    "Impressions": df["Impressions"].sum(),
    "Clicks":      df["Clicks"].sum(),
    "Leads":       df["Leads"].sum(),
    "Customers":   df["Customers"].sum(),
}
funnel_df = pd.Series(funnel_data)
funnel_colors = ["#4361ee","#7209b7","#10b981","#f59e0b"]
bars3 = ax.barh(funnel_df.index[::-1], funnel_df.values[::-1],
                color=funnel_colors[::-1], edgecolor="white", height=0.55)
ax.set_title("Overall Marketing Funnel (All Channels)", fontweight="bold")
ax.set_xlabel("Count")
for bar in bars3:
    ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
            f"{bar.get_width():,.0f}", va="center", fontsize=9)

# 3e. CPA by Channel
ax = axes[1, 1]
cpa = agg["Avg_CPA"].sort_values()
cpa_colors = [ch_colors[c] for c in cpa.index]
bars4 = ax.bar(cpa.index, cpa.values, color=cpa_colors, width=0.55, edgecolor="white")
ax.set_title("Cost Per Acquisition (CPA) by Channel", fontweight="bold")
ax.set_ylabel("Avg CPA (₹)")
ax.set_xticklabels(cpa.index, rotation=20, ha="right")
for bar in bars4:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            f"₹{bar.get_height():.0f}", ha="center", fontsize=9)

# 3f. Spend vs Revenue Scatter
ax = axes[1, 2]
for ch in channels:
    cdf = df[df["Channel"]==ch]
    ax.scatter(cdf["Spend"], cdf["Revenue"], label=ch,
               color=ch_colors[ch], s=60, alpha=0.75, edgecolors="white", linewidth=0.5)
max_val = max(df["Spend"].max(), df["Revenue"].max())
ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, label="Break-even line")
ax.set_title("Monthly Spend vs Revenue (All Channels)", fontweight="bold")
ax.set_xlabel("Spend (₹)")
ax.set_ylabel("Revenue (₹)")
ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig("marketing_funnel_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Chart saved: marketing_funnel_analysis.png")
