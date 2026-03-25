# ============================================================
# Global Superstore Sales & Profitability Analysis
# Analyst  : Fidha Nesrin M
# Tools    : Python | Pandas | NumPy | Matplotlib | Seaborn
# Dataset  : Global Superstore (2021–2024) | 2,000 Orders
# GitHub   : github.com/FidhaNesrin
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── LOAD DATA ────────────────────────────────────────────────
df = pd.read_csv('superstore_data.csv', parse_dates=['Order Date'])
df['Year']  = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.to_period('M')

print("Dataset shape:", df.shape)
print(df.dtypes)
print(df.describe())

# ── STYLE ────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
BLUE   = '#1F4E79'
TEAL   = '#2E86AB'
ORANGE = '#F4A261'
GREEN  = '#2A9D8F'
RED    = '#E76F51'

# ── 1. KPI SUMMARY ───────────────────────────────────────────
total_sales   = df['Sales'].sum()
total_profit  = df['Profit'].sum()
margin        = total_profit / total_sales * 100
total_orders  = df['Order ID'].nunique()

print(f"\n{'='*40}")
print(f"Total Sales   : ${total_sales:,.2f}")
print(f"Total Profit  : ${total_profit:,.2f}")
print(f"Profit Margin : {margin:.1f}%")
print(f"Total Orders  : {total_orders:,}")
print(f"{'='*40}\n")

# ── 2. MONTHLY REVENUE TREND ─────────────────────────────────
monthly = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly['Order Date'] = monthly['Order Date'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.fill_between(monthly['Order Date'], monthly['Sales'], alpha=0.15, color=BLUE)
ax.plot(monthly['Order Date'], monthly['Sales'], color=BLUE, linewidth=2.5, marker='o', markersize=3)
ax.set_title('Monthly Revenue Trend (2021–2024)', fontsize=14, fontweight='bold', color=BLUE)
ax.set_ylabel('Sales ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_facecolor('#F8FBFF')
plt.tight_layout()
plt.savefig('fig2_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 3. SALES & PROFIT BY REGION ──────────────────────────────
reg = df.groupby('Region')[['Sales', 'Profit']].sum().sort_values('Sales', ascending=False)

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(reg))
w = 0.38
ax.bar(x - w/2, reg['Sales'],  w, label='Sales',  color=BLUE,  alpha=0.9)
ax.bar(x + w/2, reg['Profit'], w, label='Profit', color=GREEN, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(reg.index, fontsize=11)
ax.set_title('Sales & Profit by Region', fontsize=13, fontweight='bold', color=BLUE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax.legend()
ax.set_facecolor('#F8FBFF')
plt.tight_layout()
plt.savefig('fig3_region.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 4. CATEGORY PROFITABILITY ────────────────────────────────
cat = df.groupby('Category')[['Sales', 'Profit']].sum()
cat['Margin%'] = (cat['Profit'] / cat['Sales'] * 100).round(1)
print("\nCategory Profitability:\n", cat)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = [BLUE, TEAL, ORANGE]
axes[0].bar(cat.index, cat['Sales'],   color=colors, alpha=0.9)
axes[0].set_title('Sales by Category', fontsize=12, fontweight='bold', color=BLUE)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
axes[1].bar(cat.index, cat['Margin%'], color=colors, alpha=0.9)
axes[1].set_title('Profit Margin % by Category', fontsize=12, fontweight='bold', color=BLUE)
axes[1].set_ylabel('Margin %')
for ax in axes:
    ax.set_facecolor('#F8FBFF')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_category.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 5. SUB-CATEGORY PROFIT RANKING ───────────────────────────
sub = df.groupby('Sub-Category')['Profit'].sum().sort_values()
colors_sub = [RED if v < 0 else GREEN for v in sub.values]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(sub.index, sub.values, color=colors_sub, alpha=0.88)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Profit by Sub-Category', fontsize=13, fontweight='bold', color=BLUE)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax.set_facecolor('#F8FBFF')
plt.tight_layout()
plt.savefig('fig5_subcategory.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 6. CUSTOMER SEGMENT DONUT ────────────────────────────────
seg_s = df.groupby('Segment')['Sales'].sum()

fig, ax = plt.subplots(figsize=(7, 5))
ax.pie(seg_s, labels=seg_s.index, autopct='%1.1f%%',
       colors=[BLUE, TEAL, ORANGE], startangle=140,
       wedgeprops=dict(width=0.55), pctdistance=0.75)
ax.set_title('Sales by Customer Segment', fontsize=13, fontweight='bold', color=BLUE)
plt.tight_layout()
plt.savefig('fig6_segment.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 7. DISCOUNT vs PROFIT ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
scatter_colors = [GREEN if p > 0 else RED for p in df['Profit']]
ax.scatter(df['Discount'], df['Profit'], c=scatter_colors, alpha=0.35, s=15)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Discount Rate')
ax.set_ylabel('Profit ($)')
ax.set_title('Impact of Discount on Profit', fontsize=13, fontweight='bold', color=BLUE)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.set_facecolor('#F8FBFF')
profit_patch = mpatches.Patch(color=GREEN, label='Profitable')
loss_patch   = mpatches.Patch(color=RED,   label='Loss-making')
ax.legend(handles=[profit_patch, loss_patch])
plt.tight_layout()
plt.savefig('fig7_discount.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 8. SHIPPING MODE ANALYSIS ────────────────────────────────
ship = df.groupby('Ship Mode')[['Sales', 'Profit']].sum()
ship['Margin%'] = (ship['Profit'] / ship['Sales'] * 100).round(1)
ship = ship.sort_values('Sales', ascending=False)
print("\nShipping Mode Analysis:\n", ship)

# ── 9. RFM ANALYSIS ──────────────────────────────────────────
snapshot = df['Order Date'].max() + pd.Timedelta(days=1)
rfm = df.groupby('Customer Name').agg(
    Recency   = ('Order Date', lambda x: (snapshot - x.max()).days),
    Frequency = ('Order ID',   'nunique'),
    Monetary  = ('Sales',      'sum')
).reset_index()

rfm['R'] = pd.qcut(rfm['Recency'],   4, labels=[4, 3, 2, 1]).astype(int)
rfm['F'] = pd.qcut(rfm['Frequency'], 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
rfm['M'] = pd.qcut(rfm['Monetary'],  4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
rfm['RFM_Score'] = rfm['R'] + rfm['F'] + rfm['M']

def segment_rfm(score):
    if score >= 10: return 'Champions'
    elif score >= 8: return 'Loyal'
    elif score >= 6: return 'Potential'
    else:            return 'At Risk'

rfm['Segment'] = rfm['RFM_Score'].apply(segment_rfm)
print("\nRFM Segment Distribution:\n", rfm['Segment'].value_counts())

rfm_counts = rfm['Segment'].value_counts()
colors_rfm = {'Champions': GREEN, 'Loyal': BLUE, 'Potential': TEAL, 'At Risk': RED}

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(rfm_counts.index, rfm_counts.values,
              color=[colors_rfm[s] for s in rfm_counts.index], alpha=0.9)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(int(bar.get_height())), ha='center', fontweight='bold', fontsize=11)
ax.set_title('Customer Segmentation — RFM Analysis', fontsize=13, fontweight='bold', color=BLUE)
ax.set_ylabel('Number of Customers')
ax.set_facecolor('#F8FBFF')
plt.tight_layout()
plt.savefig('fig9_rfm.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 10. TOP 10 CUSTOMERS ─────────────────────────────────────
top10 = df.groupby('Customer Name')['Sales'].sum().nlargest(10).sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(top10.index, top10.values, color=BLUE, alpha=0.85)
for bar in bars:
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
            f'${bar.get_width():,.0f}', va='center', fontsize=9)
ax.set_title('Top 10 Customers by Revenue', fontsize=13, fontweight='bold', color=BLUE)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax.set_facecolor('#F8FBFF')
plt.tight_layout()
plt.savefig('fig10_top_customers.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Analysis complete. All figures saved.")
