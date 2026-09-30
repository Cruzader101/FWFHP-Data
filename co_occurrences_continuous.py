# %% Imports and Setup
#
# Option A (continuous PHQ-2) co-occurrence analysis.
#
# The sibling script `co_occurences.py` dichotomizes the PHQ-2 score at > 2 and
# compares two buckets with Fisher's exact. That collapses the graded 0-6 scale
# (a 3 and a 6 are treated identically) and shrinks the "positive" group to ~61
# patients, which is why every diagnosis came back non-significant after FDR.
#
# This script instead treats the score as a CONTINUOUS predictor and asks, per
# diagnosis, "does the diagnosis get steadily more likely as the PHQ-2 score
# climbs?" -- a dose-response question fit with logistic regression on ALL ~1,896
# responders. `co_occurences.py` is left untouched so the two can be compared.
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from data_paths import FWFHP_EXPORT

DATA_FILE = FWFHP_EXPORT

# Column keys -> short display names (identical to co_occurences.py so the binary
# and continuous results line up row-for-row).
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
# Same load path as co_occurences.py: PHQ-2 score + diagnosis columns in one export.
df = pd.read_csv(DATA_FILE)
df = df.rename(columns={"Record ID": "record_id", "PHQ-2 Score": "a_phq2_score"})
df = df.dropna(subset=['a_phq2_score'])

# DATA_LABELS export: each diagnosis column holds "Yes"/"No" (blank = absent).
dx_cols = list(DX_MAP.keys())
for col in dx_cols:
    df[col] = (df[col] == "Yes").astype(int)

# Adults (>= 18) only, matching analyses.py and co_occurences.py: one consistent
# analytic cohort across every paper analysis.
df = df[df['Age'] >= 18]

print(f"Adult responders with a PHQ-2 score: n = {len(df)}")
print(f"PHQ-2 score distribution:\n{df['a_phq2_score'].value_counts().sort_index().to_string()}")

# %% Per-diagnosis logistic regression: Dx ~ a_phq2_score
#
# The coefficient on a_phq2_score is the log-odds change per 1-point rise in the
# PHQ-2; exp() of it is the odds ratio per point. Diagnoses with only a handful of
# total events cause perfect/quasi-complete separation and won't converge -- those
# are caught, flagged with a note, and set to NaN rather than aborting the loop.
# (Firth penalized logistic -- Option B -- is the real fix for those zero/near-zero
# cell diagnoses, but it needs an add-on package and is out of scope here.)
records = []
for col, name in DX_MAP.items():
    df['dx'] = df[col]
    n_events = int(df['dx'].sum())
    or_val = ci_lo = ci_hi = p_val = np.nan
    note = ''

    if n_events == 0:
        note = 'no events (0 cases)'
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')  # promote convergence warnings to errors
                fit = smf.logit("dx ~ a_phq2_score", data=df).fit(disp=0)
            beta = fit.params['a_phq2_score']
            ci = fit.conf_int().loc['a_phq2_score']
            or_val = np.exp(beta)
            ci_lo, ci_hi = np.exp(ci.iloc[0]), np.exp(ci.iloc[1])
            p_val = fit.pvalues['a_phq2_score']
            # Guard against silently degenerate fits (separation -> exploding CI).
            if not np.isfinite(or_val) or ci_hi > 1e6:
                or_val = ci_lo = ci_hi = p_val = np.nan
                note = 'unstable fit (separation / exploding CI)'
        except (PerfectSeparationError, Warning, np.linalg.LinAlgError) as exc:
            note = f'did not converge ({type(exc).__name__})'

    records.append({
        'Diagnosis':  name,
        'n_events':   n_events,
        'OR_per_pt':  or_val,
        'CI_low':     ci_lo,
        'CI_high':    ci_hi,
        'p_val':      p_val,
        'note':       note,
    })

df = df.drop(columns='dx')
results = pd.DataFrame(records)

# %% Benjamini-Hochberg FDR across the diagnoses that produced a valid p-value
valid = results['p_val'].notna()
results['q_val'] = np.nan
if valid.any():
    _, q_vals, _, _ = multipletests(results.loc[valid, 'p_val'], method='fdr_bh')
    results.loc[valid, 'q_val'] = q_vals

results['sig'] = results['q_val'].apply(
    lambda q: '' if pd.isna(q) else ('**' if q < 0.01 else ('*' if q < 0.05 else 'ns'))
)

# Sort strongest positive dose-response first for readability.
results = results.sort_values('OR_per_pt', ascending=False, na_position='last').reset_index(drop=True)

print("\nLogistic regression: Dx ~ PHQ-2 score (OR per 1-point increase), BH-FDR corrected\n")
print(results[['Diagnosis', 'n_events', 'OR_per_pt', 'CI_low', 'CI_high',
               'p_val', 'q_val', 'sig', 'note']].to_string(index=False, float_format='{:.4f}'.format))

# %% Chart: forest plot of per-point odds ratios
# Same forest-plot idiom as analyses.py (log x-scale, CI whisker, OR dot, dashed
# null line at OR=1, right-edge OR (CI) labels). Diagnoses without a finite OR/CI
# (no events / non-converged) are excluded from the bars and listed in a note box.
plot_df = results[results['OR_per_pt'].notna() &
                  results['CI_low'].notna() &
                  results['CI_high'].notna()].reset_index(drop=True)
excluded = results[results['OR_per_pt'].isna()]

fig, ax = plt.subplots(figsize=(9, len(plot_df) * 0.55 + 2))
y_pos = list(range(len(plot_df) - 1, -1, -1))  # top-to-bottom order

for i, (_, row) in zip(y_pos, plot_df.iterrows()):
    ax.plot([row['CI_low'], row['CI_high']], [i, i], color='steelblue', linewidth=2)
    ax.plot(row['OR_per_pt'], i, 'o', color='steelblue', markersize=8, zorder=5)
    label_tf = plt.matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    star = f"  {row['sig']}" if row['sig'] not in ('', 'ns') else ''
    ax.text(1.02, i, f"{row['OR_per_pt']:.2f} ({row['CI_low']:.2f}, {row['CI_high']:.2f}){star}",
            ha='left', va='center', fontsize=8, transform=label_tf)

ax.axvline(x=1, color='black', linestyle='--', linewidth=1)  # null effect
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df['Diagnosis'].tolist())
ax.set_xlabel('Odds Ratio per 1-point PHQ-2 increase (95% CI)', fontsize=12)
ax.set_title('Diagnosis dose-response to PHQ-2 score\n(logistic regression, per diagnosis, BH-FDR)',
             fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
ax.grid(axis='x', linestyle=':', alpha=0.5)

if len(excluded):
    lines = [f"Excluded (no stable estimate): "]
    lines += [f"  • {r['Diagnosis']}: {r['note']}" for _, r in excluded.iterrows()]
    ax.text(0.99, 0.01, '\n'.join(lines), transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

plt.tight_layout()
plt.savefig('charts/chart_continuous_or.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
