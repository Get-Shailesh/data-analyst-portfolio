"""
E-Commerce Sales Trend Analysis
================================
Dataset: Synthetically generated retail dataset (mirrors real-world patterns)
Tools: Python, Pandas, Matplotlib, Seaborn
Author: Aryan Kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. Generate Dataset ─────────────────────────────────────────────────────
np.random.seed(42)

months = pd.date_range(start="2022-01-01", periods=12, freq="MS")
categories = ["Electronics", "Clothing", "Home & Garden"]

data = {
    "Month": [],
    "Category": [],
    "Revenue": [],
    "Units_Sold": [],
    "Avg_Order_Value": [],
}

revenue_base = {
    "Electronics":   [42000,38500,45000,51000,48000,55000,61000,58000,63000,72000,89000,104000],
    "Clothing":      [28000,31000,35000,29000,33000,38000,41000,44000,39000,48000,62000,78000],
    "Home & Garden": [19000,17500,24000,31000,36000,42000,39000,35000,28000,22000,19000,21000],
}

for cat, rev_list in revenue_base.items():
    for i, month in enumerate(months):
        rev = rev_list[i] + np.random.randint(-800, 800)
        units = int(rev / np.random.uniform(80, 150))
        aov = round(rev / units, 2)
        data["Month"].append(month)
        data["Category"].append(cat)
        data["Revenue"].append(rev)
        data["Units_Sold"].append(units)
        data["Avg_Order_Value"].append(aov)

df = pd.DataFrame(data)
df.to_csv("sales_data.csv", index=False)
print("✅ Dataset created: sales_data.csv")
print(df.head(10).to_string())

# ── 2. Summary Stats ────────────────────────────────────────────────────────
print("\n📊 Revenue Summary by Category:")
summary = df.groupby("Category")["Revenue"].agg(["sum", "mean", "max", "min"])
summary.columns = ["Total Revenue", "Avg Monthly Revenue", "Peak Month", "Lowest Month"]
summary["Total Revenue"] = summary["Total Revenue"].map("₹{:,.0f}".format)
summary["Avg Monthly Revenue"] = summary["Avg Monthly Revenue"].map("₹{:,.0f}".format)
print(summary.to_string())

total_revenue = df["Revenue"].sum()
print(f"\n💰 Total Annual Revenue: ₹{total_revenue:,.0f}")

# MoM growth
monthly_total = df.groupby("Month")["Revenue"].sum()
mom_growth = monthly_total.pct_change() * 100
print(f"\n📈 Best MoM Growth: {mom_growth.max():.1f}% in {mom_growth.idxmax().strftime('%B %Y')}")
print(f"📉 Worst MoM Change: {mom_growth.min():.1f}% in {mom_growth.idxmin().strftime('%B %Y')}")

# ── 3. Visualizations ───────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("E-Commerce Sales Trend Analysis — 2022", fontsize=16, fontweight="bold", y=1.01)

colors = {"Electronics": "#4361ee", "Clothing": "#7209b7", "Home & Garden": "#10b981"}

# Plot 1: Line chart — Monthly Revenue by Category
ax1 = axes[0, 0]
for cat in categories:
    cat_df = df[df["Category"] == cat].sort_values("Month")
    ax1.plot(cat_df["Month"].dt.strftime("%b"), cat_df["Revenue"],
             marker="o", label=cat, color=colors[cat], linewidth=2.5, markersize=5)
ax1.set_title("Monthly Revenue by Category", fontweight="bold")
ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue (₹)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
ax1.legend()
ax1.tick_params(axis="x", rotation=45)

# Plot 2: Stacked Bar — Total Monthly Revenue
ax2 = axes[0, 1]
pivot = df.pivot_table(index="Month", columns="Category", values="Revenue", aggfunc="sum")
pivot.index = pivot.index.strftime("%b")
pivot.plot(kind="bar", stacked=True, ax=ax2, color=list(colors.values()), width=0.7, legend=True)
ax2.set_title("Stacked Monthly Revenue", fontweight="bold")
ax2.set_xlabel("Month")
ax2.set_ylabel("Revenue (₹)")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
ax2.tick_params(axis="x", rotation=45)

# Plot 3: Category share (Pie)
ax3 = axes[1, 0]
cat_totals = df.groupby("Category")["Revenue"].sum()
ax3.pie(cat_totals, labels=cat_totals.index, autopct="%1.1f%%",
        colors=list(colors.values()), startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax3.set_title("Revenue Share by Category", fontweight="bold")

# Plot 4: MoM Growth
ax4 = axes[1, 1]
monthly_total_df = df.groupby("Month")["Revenue"].sum().reset_index()
monthly_total_df["MoM_Growth"] = monthly_total_df["Revenue"].pct_change() * 100
bar_colors = ["#10b981" if x >= 0 else "#ef4444" for x in monthly_total_df["MoM_Growth"].fillna(0)]
ax4.bar(monthly_total_df["Month"].dt.strftime("%b"), monthly_total_df["MoM_Growth"].fillna(0),
        color=bar_colors, width=0.6, edgecolor="white")
ax4.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax4.set_title("Month-over-Month Revenue Growth (%)", fontweight="bold")
ax4.set_xlabel("Month")
ax4.set_ylabel("Growth (%)")
ax4.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("sales_trend_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Chart saved: sales_trend_analysis.png")
