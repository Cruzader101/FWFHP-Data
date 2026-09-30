# %% Imports and Setup
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

from data_paths import FWFHP_EXPORT

DEMO_FILE = FWFHP_EXPORT

df = pd.read_csv(DEMO_FILE)
df = df.rename(columns={
    "Record ID":                                        "record_id",
    "PHQ-2 Score":                                      "a_phq2_score",
    "F32.0 Major Depressive Disorder, Mild":            "dx_mdd_mild",
    "F32.1 Major Depressive Disorder, Moderate":        "dx_mdd_mod",
    "F32.2 Major Depressive Disorder, Severe":          "dx_mdd_severe",
    "F32.0 Major Depressive Disorder w/ Psychotic Features": "dx_mdd_psychotic",
    "F41.1 Generalized Anxiety Disorder":               "dx_gad",
})

df = df.dropna(subset=["a_phq2_score"])

# Adults (>= 18) only, matching analyses.py: one consistent analytic cohort across
# every paper analysis.
df = df[df["Age"] >= 18]

df["PHQ2_high"] = df["a_phq2_score"] > 2

dx_cols = ["dx_mdd_mild", "dx_mdd_mod", "dx_mdd_severe", "dx_mdd_psychotic", "dx_gad"]
for col in dx_cols:
    df[col] = df[col] == "Yes"

# %% Care Gap Calculation
high = df[df["PHQ2_high"]].copy()
high["any_mh_dx"] = high[dx_cols].any(axis=1)

n_positive    = len(high)
n_diagnosed   = int(high["any_mh_dx"].sum())
n_undiagnosed = n_positive - n_diagnosed
rate          = n_diagnosed / n_positive

ci_low, ci_high = proportion_confint(n_diagnosed, n_positive, method='beta')

print(f"PHQ-2 > 2:            n = {n_positive}")
print(f"Formally diagnosed:   {n_diagnosed} ({rate:.1%})")
print(f"Never diagnosed:      {n_undiagnosed} ({1 - rate:.1%})")
print(f"95% CI (Clopper-Pearson): [{ci_low:.1%}, {ci_high:.1%}]")

# %% Full 2x2: screening status x diagnosis
#
# The screen-positive numerator alone (2/61) implies diagnosis was screen-driven.
# It was not: most mental-health diagnoses in this cohort went to patients who
# screened NEGATIVE, so the full cross-tab is reported rather than one cell of it.
df["any_mh_dx"] = df[dx_cols].any(axis=1)

low = df[~df["PHQ2_high"]]
n_neg           = len(low)
n_neg_diagnosed = int(low["any_mh_dx"].sum())
neg_rate        = n_neg_diagnosed / n_neg
neg_lo, neg_hi  = proportion_confint(n_neg_diagnosed, n_neg, method='beta')

n_dx_total = int(df["any_mh_dx"].sum())

print("\n--- Screening status x depressive/anxiety diagnosis ---")
print(pd.crosstab(df["PHQ2_high"], df["any_mh_dx"],
                  rownames=["PHQ-2 > 2"], colnames=["any MH dx"]).to_string())
print(f"\nTotal MH diagnoses in adult cohort: {n_dx_total}")
print(f"  among screen-positive: {n_diagnosed} ({n_diagnosed / n_dx_total:.1%} of all dx)")
print(f"  among screen-negative: {n_neg_diagnosed} ({n_neg_diagnosed / n_dx_total:.1%} of all dx)")
print(f"Screen-negative diagnosis rate: {n_neg_diagnosed}/{n_neg} = {neg_rate:.3%} "
      f"(95% CI {neg_lo:.3%}-{neg_hi:.3%})")

# All three screen-negative diagnoses sit at PHQ-2 = 2, one point under the cutoff;
# that is a threshold artifact, not diagnosis unrelated to symptoms.
print("\nPHQ-2 scores of screen-negative diagnosed patients:")
print(low.loc[low["any_mh_dx"], "a_phq2_score"].value_counts().sort_index().to_string())

print("\nCounts by diagnostic code:")
for col in dx_cols:
    print(f"  {col:20s} {int(df[col].sum())}")

# %% Care Gap Chart
fig, ax = plt.subplots(figsize=(9, 3.5))

ax.barh(0, n_diagnosed,   color='#4a9e6b', height=0.5)
ax.barh(0, n_undiagnosed, left=n_diagnosed, color='#e05c4b', height=0.5)

ax.text(n_diagnosed / 2, 0,
        f'Diagnosed\nn={n_diagnosed} ({rate:.1%})',
        ha='center', va='center', fontsize=11, color='white', fontweight='bold')
ax.text(n_diagnosed + n_undiagnosed / 2, 0,
        f'Never Diagnosed\nn={n_undiagnosed} ({1 - rate:.1%})',
        ha='center', va='center', fontsize=11, color='white', fontweight='bold')

ax.set_xlim(0, n_positive)
ax.set_yticks([])
ax.set_xlabel('Number of Patients', fontsize=12)
ax.set_title(
    f'Mental Health Care Gap Among PHQ-2 Positive Patients (n={n_positive})\n'
    f'Formal diagnosis rate: {rate:.1%}  (95% CI: {ci_low:.1%}–{ci_high:.1%})',
    fontsize=13, fontweight='bold'
)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#4a9e6b', label=f'Formally Diagnosed (n={n_diagnosed})'),
    Patch(color='#e05c4b', label=f'Never Diagnosed (n={n_undiagnosed})'),
], loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('charts/chart_care_gap.png', dpi=150)
plt.show()
