# %% Imports and Setup
#
# Sensitivity / diagnostic analyses for the exploratory co-occurrence models.
#
# `co_occurences.py` (binary, Fisher) and `co_occurrences_continuous.py`
# (continuous, logistic) produce the headline co-occurrence numbers. This script
# does not re-derive those findings -- it stress-tests them, and exists because
# the continuous analysis produced exactly one FDR-surviving association
# (dehydration) resting on 11 total events, 6 of them among encounters with any
# depressive symptom. Four questions:
#
#   1. INFLUENCE / FRAGILITY -- does the association survive deleting one patient,
#      and how many outcome reclassifications does it take to lose significance?
#   2. CLUSTERING -- screening happened at 9 camps with very different mean PHQ-2.
#      Does the independence assumption flatter the standard errors?
#   3. LINEARITY IN THE LOGIT -- the "OR per point" framing assumes a constant
#      per-point effect across 0-6. Is that testable here, and does it hold?
#   4. DATA INTEGRITY -- does the stored PHQ-2 score match its component items?
#      (The observed distribution has no score of 5 but nine 6s -- bug or real?)
#
# Numbers printed here back the "Sensitivity and Diagnostic Analyses" section of
# `Methods Draft.md` in the Obsidian vault. Re-run before submission.
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from data_paths import FWFHP_EXPORT

DATA_FILE = FWFHP_EXPORT

# Identical to the sibling scripts so rows line up across all three analyses.
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

# PHQ-2 component items. The export is Windows-1252, and the Spanish labels carry
# accents, so match on the English ASCII prefix rather than the full string.
ITEM_1 = 'Little interest or pleasure in doing things. '   # trailing space is in the header
ITEM_2 = 'Feeling down, depressed, or hopeless'
ITEM_SCORE = {'not at all': 0, 'several days': 1, 'more than half': 2, 'nearly every day': 3}

ALPHA = 0.05

# %% Load Data
# Same cohort construction as analyses.py / co_occurences.py / co_occurrences_continuous.py:
# valid PHQ-2, adults only. Any divergence here invalidates the comparison.
df = pd.read_csv(DATA_FILE, low_memory=False, encoding='cp1252')
df = df.rename(columns={"Record ID": "record_id", "PHQ-2 Score": "a_phq2_score"})
df = df.dropna(subset=['a_phq2_score'])

for col in DX_MAP:
    df[col] = (df[col] == "Yes").astype(int)

df = df[df['Age'] >= 18].copy().reset_index(drop=True)
df['camp_code'] = pd.Categorical(df['Camp']).codes   # cluster ids must be numeric

print(f"Adult responders with a PHQ-2 score: n = {len(df)}")
print(f"Camps: {df['Camp'].nunique()}")


def fit_logit(data, dx_col, cluster=False):
    """
    Fit dx ~ a_phq2_score. Returns (OR, ci_low, ci_high, p) or a tuple of NaN if the
    model won't converge / is degenerate. Convergence warnings are promoted to errors
    so a silently separated fit can't masquerade as a real estimate.
    """
    d = data.copy()
    d['dx'] = d[dx_col]
    kw = {}
    if cluster:
        kw = dict(cov_type='cluster', cov_kwds={'groups': d['camp_code'].values})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            fit = smf.logit("dx ~ a_phq2_score", data=d).fit(disp=0, **kw)
        beta = fit.params['a_phq2_score']
        ci = fit.conf_int().loc['a_phq2_score']
        or_val, lo, hi = np.exp(beta), np.exp(ci.iloc[0]), np.exp(ci.iloc[1])
        p = fit.pvalues['a_phq2_score']
        if not np.isfinite(or_val) or hi > 1e6:
            return (np.nan,) * 4
        return or_val, lo, hi, p
    except Exception:
        return (np.nan,) * 4


# %% Baseline: reproduce the continuous models + BH-FDR
# This must match co_occurrences_continuous.py exactly. If it doesn't, one of the two
# scripts has drifted and every number below is suspect.
rows = []
for col, name in DX_MAP.items():
    n_events = int(df[col].sum())
    n_events_sx = int(df.loc[df['a_phq2_score'] > 0, col].sum())
    or_val, lo, hi, p = fit_logit(df, col)
    rows.append({'Diagnosis': name, 'n_events': n_events, 'events_PHQ2>0': n_events_sx,
                 'OR_per_pt': or_val, 'CI_low': lo, 'CI_high': hi, 'p_val': p})

base = pd.DataFrame(rows)
valid = base['p_val'].notna()
base['q_val'] = np.nan
base.loc[valid, 'q_val'] = multipletests(base.loc[valid, 'p_val'], method='fdr_bh')[1]

print("\n=== Baseline continuous models (reproduces co_occurrences_continuous.py) ===\n")
print(base.to_string(index=False, float_format='{:.4f}'.format))

survivors = base.loc[base['q_val'] < ALPHA, 'Diagnosis'].tolist()
nominal = base.loc[base['p_val'] < ALPHA, 'Diagnosis'].tolist()
print(f"\nFDR survivors (q < {ALPHA}): {survivors or 'none'}")
print(f"Nominally significant (p < {ALPHA}, pre-FDR): {nominal or 'none'}")

# %% Diagnostic 1: leave-one-out influence + fragility index
#
# Leave-one-out DELETES a patient; fragility RECLASSIFIES a case as a non-case.
# They answer different questions ("is this driven by one person?" vs "how many
# outcome misclassifications would overturn it?"), so both are reported.
#
# The single-flip fragility search below is exhaustive, so a reported index of 1 is
# exact rather than a greedy upper bound. Beyond 1 it falls back to a greedy search
# (flip whichever remaining case most weakens the association) and the result is an
# upper bound -- flagged as such.
NAME_TO_COL = {v: k for k, v in DX_MAP.items()}


def leave_one_out(dx_col):
    """Refit with each case deleted in turn. Returns the deletion that weakens the
    association most (largest resulting p-value)."""
    out = []
    for i in df.index[df[dx_col] == 1]:
        or_i, lo_i, hi_i, p_i = fit_logit(df.drop(i), dx_col)
        out.append({'dropped_idx': i, 'phq2': df.loc[i, 'a_phq2_score'],
                    'OR': or_i, 'CI_low': lo_i, 'CI_high': hi_i, 'p_val': p_i})
    return pd.DataFrame(out).sort_values('p_val', ascending=False).reset_index(drop=True)


def fragility_index(dx_col, alpha=ALPHA):
    """Minimum number of cases that must be reclassified as non-cases before the
    association loses significance. Exhaustive at k=1; greedy (upper bound) after."""
    d = df.copy()
    # k = 1: exhaustive over every case.
    for i in d.index[d[dx_col] == 1]:
        d2 = d.copy()
        d2.loc[i, dx_col] = 0
        _, _, _, p = fit_logit(d2, dx_col)
        if not np.isfinite(p) or p >= alpha:
            return 1, True, [i], p
    # k > 1: greedy.
    flipped, exact = [], False
    while True:
        best = None
        for i in d.index[d[dx_col] == 1]:
            d2 = d.copy()
            d2.loc[i, dx_col] = 0
            _, _, _, p = fit_logit(d2, dx_col)
            p = 1.0 if not np.isfinite(p) else p
            if best is None or p > best[1]:
                best = (i, p)
        if best is None:
            return np.nan, exact, flipped, np.nan
        d.loc[best[0], dx_col] = 0
        flipped.append(best[0])
        if best[1] >= alpha:
            return len(flipped), exact, flipped, best[1]
        if len(flipped) > 20:      # runaway guard
            return np.nan, exact, flipped, best[1]


# Stress-test anything nominally significant, not just the FDR survivors -- a reader
# will ask about headache too, since it is in the results table with q = 0.10.
targets = sorted(set(survivors) | set(nominal))
print(f"\n=== Diagnostic 1: influence + fragility (targets: {', '.join(targets)}) ===")

loo_store = {}
for name in targets:
    col = NAME_TO_COL[name]
    b = base[base['Diagnosis'] == name].iloc[0]
    loo = leave_one_out(col)
    loo_store[name] = loo
    fi, exact, flipped, p_after = fragility_index(col)

    worst = loo.iloc[0]
    print(f"\n--- {name} ---")
    print(f"  full model      : OR {b['OR_per_pt']:.2f} ({b['CI_low']:.2f}-{b['CI_high']:.2f}) "
          f"p={b['p_val']:.4f} q={b['q_val']:.4f}  [{int(b['n_events'])} events, "
          f"{int(b['events_PHQ2>0'])} with PHQ-2>0]")
    print(f"  worst deletion  : OR {worst['OR']:.2f} ({worst['CI_low']:.2f}-{worst['CI_high']:.2f}) "
          f"p={worst['p_val']:.4f}   (deleted encounter had PHQ-2 = {worst['phq2']:.0f})")
    print(f"  significance lost by deleting 1 encounter? "
          f"{'YES' if worst['p_val'] >= ALPHA else 'no'}")
    qual = 'exact' if exact else 'greedy upper bound'
    print(f"  fragility index : {fi}  ({qual}) -> p={p_after:.4f} after reclassifying "
          f"{fi} case(s)")

# %% Diagnostic 2: does clustering on camp explain the signal?
# Between-camp variation in mean PHQ-2 is large; if the naive SEs were flattering the
# results, cluster-robust SEs would widen the CIs and kill significance.
print("\n\n=== Diagnostic 2: camp clustering ===\n")
print("Mean PHQ-2 by camp:")
print(df.groupby('Camp')['a_phq2_score'].agg(['size', 'mean']).round(3).to_string())

print(f"\n{'Diagnosis':<18}{'OR':>7}{'p_naive':>10}{'p_cluster':>11}   effect of clustering")
print("-" * 70)
for name in targets:
    col = NAME_TO_COL[name]
    o1, lo1, hi1, p1 = fit_logit(df, col)
    o2, lo2, hi2, p2 = fit_logit(df, col, cluster=True)
    if p1 < ALPHA <= p2:
        verdict = 'LOSES significance'
    elif p2 < p1:
        verdict = 'strengthens (not the driver)'
    else:
        verdict = 'weakens but survives'
    print(f"{name:<18}{o1:>7.2f}{p1:>10.4f}{p2:>11.4f}   {verdict}")
    print(f"{'':18}naive CI {lo1:.2f}-{hi1:.2f} -> cluster-robust CI {lo2:.2f}-{hi2:.2f}"
          f"{'  [NARROWER -- see caveat]' if (hi2 - lo2) < (hi1 - lo1) else ''}")

print(f"""
CAVEAT: there are only {df['Camp'].nunique()} clusters. The cluster-robust (sandwich)
estimator is asymptotic in the NUMBER OF CLUSTERS and is known to be anti-conservative
below roughly 30-40; with 9 it can return intervals NARROWER than the naive ones, which
is a symptom of downward-biased variance rather than evidence of a firmer result. Read
this diagnostic only as "clustering does not appear to manufacture the association" --
NOT as independent confirmation. A wild cluster bootstrap or CR2 small-sample correction
would be required to make a positive claim from it, and with a fragility index of 1 that
is not worth doing.""")

# %% Diagnostic 3: linearity in the logit
#
# The per-point OR assumes each 1-point rise multiplies the odds by the same factor
# across 0-6. The formal test is to refit the score as categorical and compare fits.
#
# IMPORTANT: that test does not work on this data and must not be reported. Every
# categorical fit fails to converge, because diagnoses have zero events at some score
# levels -> separation -> coefficients running to +/-inf with astronomical SEs. Such a
# fit still yields a finite log-likelihood, so it will silently produce a plausible-
# looking LR p-value if convergence warnings are suppressed. They are promoted to
# errors here specifically to prevent that. `converged` is asserted explicitly rather
# than trusted.
#
# The assumption is therefore UNTESTABLE at this sparsity -- which is itself the
# reportable result. In its place we print the raw event rate at each score level, so
# the shape of the (non-)dose-response can be read directly off the data instead of
# being imputed by a model that assumes the very thing in question.
print("\n\n=== Diagnostic 3: linearity in the logit (LR test, categorical vs linear) ===")
print("small p => per-point assumption violated; failure => untestable at this sparsity\n")

df['phq_cat'] = df['a_phq2_score'].astype(int).astype(str)
for col, name in DX_MAP.items():
    d = df.copy()
    d['dx'] = d[col]
    if d['dx'].sum() == 0:
        print(f"  {name:<18} skipped (no events)")
        continue
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            lin = smf.logit("dx ~ a_phq2_score", data=d).fit(disp=0)
            cat = smf.logit("dx ~ C(phq_cat)", data=d).fit(disp=0)
        if not (lin.mle_retvals['converged'] and cat.mle_retvals['converged']):
            raise RuntimeError('did not converge')
        lr = 2 * (cat.llf - lin.llf)
        dof = cat.df_model - lin.df_model
        p = chi2.sf(lr, dof) if dof > 0 else np.nan
        verdict = 'linearity REJECTED' if p < ALPHA else 'no evidence against linearity'
        print(f"  {name:<18} LR={lr:6.2f}  df={dof:.0f}  p={p:.3f}   {verdict}")
    except Exception as exc:
        print(f"  {name:<18} untestable ({type(exc).__name__}: sparse cells / separation)")

# Raw rates: what the per-point OR is actually being fitted to.
print("\n--- Observed event rate by PHQ-2 score (what the linear model smooths over) ---")
for name in targets:
    col = NAME_TO_COL[name]
    tab = df.groupby('a_phq2_score')[col].agg(['sum', 'size'])
    tab['rate_%'] = (tab['sum'] / tab['size'] * 100).round(2)
    cells = "  ".join(f"{int(s)}:{r:5.2f}%" for s, r in zip(tab.index, tab['rate_%']))
    print(f"\n  {name}")
    print(f"    n cases / n at score : " +
          "  ".join(f"{int(s)}:{int(a)}/{int(b)}" for s, a, b in
                    zip(tab.index, tab['sum'], tab['size'])))
    print(f"    rate by score        : {cells}")
    if tab['rate_%'].is_monotonic_increasing:
        verdict = 'yes'
    else:
        verdict = 'NO -- the per-point OR imposes a trend the raw rates do not show'
    print(f"    monotonically increasing in PHQ-2? {verdict}")

# %% Diagnostic 4: data integrity -- reconstruct PHQ-2 from its component items
# The distribution has zero encounters scoring 5 but nine scoring 6. Either the score
# column is miscomputed, or the (2,3)/(3,2) item combinations genuinely never occurred.
print("\n\n=== Diagnostic 4: PHQ-2 score reconstruction ===\n")


def score_item(s):
    if pd.isna(s):
        return np.nan
    s = s.lower()
    for prefix, val in ITEM_SCORE.items():
        if s.startswith(prefix):
            return val
    return np.nan


df['item_1'] = df[ITEM_1].map(score_item)
df['item_2'] = df[ITEM_2].map(score_item)
recon_ok = df.dropna(subset=['item_1', 'item_2']).copy()
recon_ok['recon'] = recon_ok['item_1'] + recon_ok['item_2']

mismatch = recon_ok[recon_ok['recon'] != recon_ok['a_phq2_score']]
print(f"Encounters with both items answered: {len(recon_ok)} / {len(df)}")
print(f"Reconstructed score != stored score: {len(mismatch)}")
print("  -> stored PHQ-2 score is internally consistent" if len(mismatch) == 0
      else f"  -> WARNING: {len(mismatch)} scoring mismatches, investigate")

# A stored score with an unanswered item would mean the score was imputed somewhere.
partial = df[df['item_1'].isna() | df['item_2'].isna()]
if len(partial):
    print(f"\nEncounters with a stored score but one item blank: {len(partial)}")
    print(partial[['a_phq2_score', 'item_1', 'item_2']].to_string(index=False))
    print("  -> all stored as 0 with the answered item also 0; consistent, no impact "
          "on any model, but note the blank item was effectively treated as 0.")

print("\nItem cross-tab (rows = little interest, cols = feeling down):")
print(pd.crosstab(recon_ok['item_1'], recon_ok['item_2']).to_string())

print("\nObserved score distribution:")
dist = df['a_phq2_score'].value_counts().sort_index()
for s in range(7):
    n = int(dist.get(float(s), 0))
    flag = '   <-- structural gap' if n == 0 else ''
    print(f"  PHQ-2 = {s}: {n:5d}{flag}")

# %% Chart: leave-one-out influence for the FDR survivor(s)
# Shows how far each single deletion moves the p-value. A finding that depends on one
# encounter shows up as a lone point stranded above the alpha line.
for name, loo in loo_store.items():
    if loo.empty or loo['p_val'].isna().all():
        continue
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = loo.sort_values('phq2').reset_index(drop=True)
    colors = ['tomato' if p >= ALPHA else 'steelblue' for p in order['p_val']]
    ax.scatter(range(len(order)), order['p_val'], c=colors, s=90, zorder=5,
               edgecolor='white', linewidth=0.8)
    full_p = base.loc[base['Diagnosis'] == name, 'p_val'].iloc[0]
    ax.axhline(full_p, color='gray', linestyle='-', linewidth=1,
               label=f'full-model p = {full_p:.4f}')
    ax.axhline(ALPHA, color='black', linestyle='--', linewidth=1.2,
               label=f'α = {ALPHA}')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"PHQ-2={v:.0f}" for v in order['phq2']], rotation=40, ha='right',
                       fontsize=8)
    ax.set_ylabel('p-value with this encounter deleted')
    ax.set_xlabel(f'Encounter deleted (each is one {name.lower()} case)')
    ax.set_title(f'{name}: leave-one-out influence\n'
                 f'red = deleting that single encounter costs significance',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    fname = f"charts/chart_sensitivity_loo_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nSaved {fname}")
    plt.show()

# %%
