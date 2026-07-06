# %% Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

DATA_FILE = r'Night Camp Data\FarmWorkerFamilyHeal-2426FWFHPDataAnalysi_DATA_LABELS_2026-07-06_1745.csv'

# Column keys

DX_MAP = {
    'E86.0 Dehydration'          : 'Dehydration',
    'R03.0 Elevated Blood Pressure': 'Elevated BP',
    'R73 Elevated Glucose'       : 'Elevated Glucose',
    'H57.10 Eye Pain'            : 'Eye Pain',
    'R53.8 Fatigue'              : 'Fatigue',
    'R51Headache, tension'       : 'Headache',
    'F51.01 Insomnia Disorder'   : 'Insomnia',
    'R21 Milaria (heat rash)'    : 'Milaria',
    'M79.1 Myalgia'              : 'Myalgia',
    'E66.9 Obesity'              : 'Obesity',
    'Z20.2 STIs'                 : 'STIs',
    'L55.9 Sunburn'              : 'Sunburn',
}

# %% Load Data
# PHQ-2 score and all diagnosis columns now live in one unified export.
df = pd.read_csv(DATA_FILE)
df = df.rename(columns={"Record ID": "record_id", "PHQ-2 Score": "a_phq2_score"})
df = df.dropna(subset=['a_phq2_score'])

# DATA_LABELS export: each diagnosis column holds the text "Yes"/"No" (blank = absent).
dx_cols = list(DX_MAP.keys())
for col in dx_cols:
    df[col] = (df[col] == "Yes").astype(int)

df['high_phq2'] = df['a_phq2_score'] > 2

high = df[df['high_phq2']]
low  = df[~df['high_phq2']]

print(f"PHQ-2 > 2 group : n = {len(high)}")
print(f"PHQ-2 <= 2 group: n = {len(low)}")

# %% Calculate Conditional Probabilities and Odds Ratios
def compute_or(n_dx_high, n_high, n_dx_low, n_low):
    """
    Odds ratio with a note when undefined.
    Returns (OR, note) where note is '' / 'OR=0' / 'OR=inf' / 'undefined'.
    """
    p_h = n_dx_high / n_high
    p_l = n_dx_low  / n_low
    if p_h == 0 and p_l == 0:
        return np.nan, 'undefined (0 cases in both groups)'
    if p_l == 0:
        return np.inf, 'OR=∞ (0 cases in low group)'
    if p_h == 0:
        return 0.0, ''
    odds_h = p_h / (1 - p_h)
    odds_l = p_l / (1 - p_l)
    return odds_h / odds_l, ''

records = []
for col, name in DX_MAP.items():
    n_high = len(high)
    n_low  = len(low)
    n_dx_high = high[col].sum()
    n_dx_low  = low[col].sum()
    p_high = n_dx_high / n_high
    p_low  = n_dx_low  / n_low
    or_val, note = compute_or(n_dx_high, n_high, n_dx_low, n_low)
    _, p_val = fisher_exact([[n_dx_high, n_high - n_dx_high],
                             [n_dx_low,  n_low  - n_dx_low]])
    records.append({
        'Diagnosis':       name,
        'n (PHQ2>2)':      int(n_dx_high),
        'n (PHQ2<=2)':     int(n_dx_low),
        'P(Dx|PHQ2>2)':   p_high,
        'P(Dx|PHQ2<=2)':  p_low,
        'OR':              or_val,
        'OR note':         note,
        'p_val':           p_val,
    })

results = pd.DataFrame(records)

# Benjamini-Hochberg FDR correction across all 12 tests
reject, q_vals, _, _ = multipletests(results['p_val'], method='fdr_bh')
results['q_val'] = q_vals
results['sig']   = results['q_val'].apply(lambda q: '**' if q < 0.01 else ('*' if q < 0.05 else 'ns'))

print(results[['Diagnosis','n (PHQ2>2)','n (PHQ2<=2)','P(Dx|PHQ2>2)','P(Dx|PHQ2<=2)','OR','p_val','q_val','sig']].to_string(index=False, float_format='{:.4f}'.format))

# %% Chart 1: P(Diagnosis | PHQ-2 > 2)
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(results['Diagnosis'], results['P(Dx|PHQ2>2)'],
       color='steelblue', edgecolor='white', linewidth=0.6)
ax.set_title(f'P(Diagnosis | PHQ-2 Score > 2)  [n = {len(high)}]',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Diagnosis')
ax.set_ylabel('Conditional Probability')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
plt.xticks(rotation=40, ha='right')
plt.tight_layout()
plt.savefig('charts/chart1_p_high_phq2.png', dpi=150)
plt.show()

# %% Chart 2: P(Diagnosis | PHQ-2 > 2) vs P(Diagnosis | PHQ-2 <= 2)
x     = np.arange(len(results))
width = 0.38

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - width / 2, results['P(Dx|PHQ2>2)'],  width,
       label=f'PHQ-2 > 2  (n={len(high)})',  color='tomato',    edgecolor='white', linewidth=0.6)
ax.bar(x + width / 2, results['P(Dx|PHQ2<=2)'], width,
       label=f'PHQ-2 ≤ 2  (n={len(low)})',  color='steelblue', edgecolor='white', linewidth=0.6)
ax.set_title('P(Diagnosis | PHQ-2 Score): High vs Low PHQ-2',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Diagnosis')
ax.set_ylabel('Conditional Probability')
ax.set_xticks(x)
ax.set_xticklabels(results['Diagnosis'], rotation=40, ha='right')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
ax.legend(fontsize=11)
for xi, (_, row) in zip(x, results.iterrows()):
    if row['sig'] != 'ns':
        max_h = max(row['P(Dx|PHQ2>2)'], row['P(Dx|PHQ2<=2)'])
        ax.text(xi, max_h + 0.003, row['sig'], ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/chart2_comparison.png', dpi=150)
plt.show()

# %% Chart 3: Odds Ratios
# Exclude undefined (0 cases in both groups) and infinite ORs from the main bars;
# annotate them separately.

plot_df  = results[results['OR'].notna() & (results['OR'] != np.inf)].reset_index(drop=True)
inf_diag = results[results['OR'] == np.inf]['Diagnosis'].tolist()
nan_diag = results[results['OR'].isna()]['Diagnosis'].tolist()

fig, ax = plt.subplots(figsize=(11, 6))
colors = ['tomato' if v > 1 else 'steelblue' for v in plot_df['OR']]
ax.bar(plot_df['Diagnosis'], plot_df['OR'],
       color=colors, edgecolor='white', linewidth=0.6)
ax.axhline(1, color='black', linestyle='--', linewidth=1.3, label='OR = 1  (no association)')
ax.set_title('Odds Ratio: Diagnosis in PHQ-2 > 2 vs PHQ-2 ≤ 2',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Diagnosis')
ax.set_ylabel('Odds Ratio')
plt.xticks(rotation=40, ha='right')
ax.legend(fontsize=11)
for bar, (_, row) in zip(ax.patches, plot_df.iterrows()):
    if row['sig'] != 'ns':
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(row['sig']), ha='center', va='bottom', fontsize=12, fontweight='bold')

notes = []
if inf_diag:
    notes.append(f'OR = ∞ (0 cases in PHQ-2 ≤ 2 group): {", ".join(inf_diag)}')
if nan_diag:
    notes.append(f'OR undefined (0 cases in both groups): {", ".join(nan_diag)}')
if notes:
    ax.text(0.99, 0.99, '\n'.join(notes),
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

plt.tight_layout()
plt.savefig('charts/chart3_odds_ratios.png', dpi=150)
plt.show()

# %% Chart 4: Raw Counts by Diagnosis (PHQ-2 > 2 vs PHQ-2 ≤ 2)
x     = np.arange(len(results))
width = 0.38

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - width / 2, results['n (PHQ2>2)'],  width,
       label=f'PHQ-2 > 2  (n={len(high)})',  color='tomato',    edgecolor='white', linewidth=0.6)
ax.bar(x + width / 2, results['n (PHQ2<=2)'], width,
       label=f'PHQ-2 ≤ 2  (n={len(low)})',  color='steelblue', edgecolor='white', linewidth=0.6)
ax.set_title('Count of Diagnoses: PHQ-2 > 2 vs PHQ-2 ≤ 2',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Diagnosis')
ax.set_ylabel('Number of Patients')
ax.set_xticks(x)
ax.set_xticklabels(results['Diagnosis'], rotation=40, ha='right')
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('charts/chart4_counts.png', dpi=150)
plt.show()
