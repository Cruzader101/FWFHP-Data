# %%

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from scipy.stats import norm

from data_paths import FWFHP_EXPORT

NHANES_data = pd.read_csv("./NHANES/df_Hispanic_NHANES.csv")
FWFHP_data = pd.read_csv(FWFHP_EXPORT)
FWFHP_data = FWFHP_data.rename(columns={
    "Record ID":              "record_id",
    "PHQ-2 Score":            "a_phq2_score",
    "Camp":                   "camp",
    "Age":                    "a_age",
    "Sex":                    "gender",
})

# NOTE: "Z13.89 Drug Use Screen" only records whether a drug screen was
# performed, not the finding -- using it as "drug use" produced a spurious
# care-process association. Use the actual result instead. Blank = not screened
# is treated as "no documented positive screen" so this covariate does not drop
# ~900 rows from the model; the predictor is "positive drug screen on record".
FWFHP_data["drug_use"] = np.where(
    FWFHP_data["Drug Use Screen Result"] == "Yes", "Yes", "No"
)

# %%

#Adding binary column for PHQ-2 score greater than 2 for both datasets

# Keep only valid PHQ-2 responders; otherwise a missing score is silently binarized
# to 0 and inflates the "negative" group (and the denominator) in both cohorts.
NHANES_data = NHANES_data.dropna(subset=["phq2_score"])
FWFHP_data  = FWFHP_data.dropna(subset=["a_phq2_score"])

# Age-match the two cohorts: NHANES only screens adults (18+) for depression, so
# restrict both sides to 18+. This makes the FWFHP denominator comparable and
# defines the NHANES adult Hispanic domain used for the weighted estimate below.
NHANES_data = NHANES_data[NHANES_data["RIDAGEYR"] >= 18]
FWFHP_data  = FWFHP_data[FWFHP_data["a_age"] >= 18]

NHANES_data["PHQ2_binary"] = (NHANES_data["phq2_score"] > 2).astype(int)
FWFHP_data["PHQ2_binary"] = (FWFHP_data["a_phq2_score"] > 2).astype(int)

# %%

# Odds Ratio Analyses Variables

FWFHP_yes = FWFHP_data[FWFHP_data["PHQ2_binary"] == 1] #instances where PHQ-2 score is greater than 2
FWFHP_no = FWFHP_data[FWFHP_data["PHQ2_binary"] == 0]
print("FWFHP Yes count:", len(FWFHP_yes))
print("FWFHP No count:", len(FWFHP_no))


NHANES_yes = NHANES_data[NHANES_data["PHQ2_binary"] == 1]
NHANES_no = NHANES_data[NHANES_data["PHQ2_binary"] == 0]
print("NHANES Yes count:", len(NHANES_yes))
print("NHANES No count:", len(NHANES_no))



# %%

# Design-based weighted NHANES prevalence
#
# NHANES is a complex, multistage probability sample. A raw (unweighted) rate
# over-represents oversampled subgroups and understates the true variance. The
# defensible comparison uses the survey weight (WTMEC2YR) for the point estimate
# and Taylor linearization over strata (SDMVSTRA) / PSUs (SDMVPSU) for the SE.

def weighted_prop_design(y, w, strata, psu):
    """Weighted proportion and Taylor-linearized design-based SE for a survey
    subpopulation (with-replacement PSU approximation, per NHANES guidance)."""
    y = np.asarray(y, float); w = np.asarray(w, float)
    strata = np.asarray(strata); psu = np.asarray(psu)
    W = w.sum()
    p = np.sum(w * y) / W
    u = w * (y - p) / W          # linearized residual contribution to p
    var = 0.0
    for h in np.unique(strata):
        hmask = strata == h
        psu_totals = np.array([u[hmask & (psu == j)].sum()
                               for j in np.unique(psu[hmask])])
        n_h = len(psu_totals)
        if n_h < 2:
            continue             # single-PSU stratum: no between-PSU variance
        var += n_h / (n_h - 1) * np.sum((psu_totals - psu_totals.mean()) ** 2)
    return p, np.sqrt(var)

p_nhanes, se_nhanes = weighted_prop_design(
    NHANES_data["PHQ2_binary"].values, NHANES_data["WTMEC2YR"].values,
    NHANES_data["SDMVSTRA"].values,    NHANES_data["SDMVPSU"].values)

# Logit-transformed CI so the interval stays inside (0, 1)
logit_n    = np.log(p_nhanes / (1 - p_nhanes))
se_logit_n = se_nhanes / (p_nhanes * (1 - p_nhanes))
ci_nhanes  = (1 / (1 + np.exp(-(logit_n - 1.96 * se_logit_n))),
              1 / (1 + np.exp(-(logit_n + 1.96 * se_logit_n))))
print(f"NHANES weighted PHQ-2 positive: {p_nhanes:.4f} "
      f"(95% CI: {ci_nhanes[0]:.4f}-{ci_nhanes[1]:.4f}); design SE = {se_nhanes:.4f}; "
      f"unweighted n = {len(NHANES_data)}")

# %%

# Odds ratio: FWFHP (clinic census, binomial) vs NHANES (survey-weighted population)
#
# FWFHP is a full clinic sample -> simple binomial variance for its logit.
# NHANES logit variance comes from the design-based SE above (delta method).
# The two logits are independent, so the log-OR variance is their sum.
p_fwfhp     = len(FWFHP_yes) / (len(FWFHP_yes) + len(FWFHP_no))
logit_f     = np.log(len(FWFHP_yes) / len(FWFHP_no))
var_logit_f = 1 / len(FWFHP_yes) + 1 / len(FWFHP_no)

log_or     = logit_f - logit_n
se_logor   = np.sqrt(var_logit_f + se_logit_n ** 2)
odds_ratio = np.exp(log_or)
or_lower   = np.exp(log_or - 1.96 * se_logor)
or_upper   = np.exp(log_or + 1.96 * se_logor)

z_stat       = log_or / se_logor
p_nhanes_cmp = 2 * norm.sf(abs(z_stat))
print(f"FWFHP PHQ-2 positive: {p_fwfhp:.4f} (n = {len(FWFHP_yes) + len(FWFHP_no)})")
print(f"OR (design-aware): {odds_ratio:.3f} (95% CI: {or_lower:.3f}-{or_upper:.3f})")
print(f"Design-aware z-test: z = {z_stat:.3f}, p = {p_nhanes_cmp:.4e}")

# NHANES comparison chart with CI (FWFHP: Wilson binomial; NHANES: design-based)
ci_fwfhp = proportion_confint(len(FWFHP_yes), len(FWFHP_yes) + len(FWFHP_no), method='wilson')

fig, ax = plt.subplots(figsize=(7, 5))
groups = ['FWFHP\n(Migrant Farmworkers)', 'NHANES\n(Hispanic Adults, weighted)']
props  = [p_fwfhp, p_nhanes]
yerr   = [[p_fwfhp - ci_fwfhp[0], p_nhanes - ci_nhanes[0]],
          [ci_fwfhp[1] - p_fwfhp, ci_nhanes[1] - p_nhanes]]
bars = ax.bar(groups, props, color=['tomato', 'steelblue'], edgecolor='white', linewidth=0.6, width=0.5)
ax.errorbar(groups, props, yerr=yerr, fmt='none', color='black', capsize=6, linewidth=1.5)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
ax.set_ylabel('PHQ-2 Positive Rate (%)')
p_label = "p < 0.001" if p_nhanes_cmp < 0.001 else f"p = {p_nhanes_cmp:.4f}"
ax.set_title(
    f'PHQ-2 Positive Prevalence: FWFHP vs. NHANES Hispanic Adults\n'
    f'OR = {odds_ratio:.2f} (95% CI: {or_lower:.2f}-{or_upper:.2f}), {p_label}  (NHANES survey-weighted)',
    fontsize=11, fontweight='bold'
)
for bar, prop in zip(bars, props):
    ax.text(bar.get_x() + bar.get_width() / 2, prop + 0.002,
            f'{prop:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/chart_nhanes_ci.png', dpi=150)
plt.show()

# %%

# Significance Testing

# %%

# Fixing missing data, causing singular matrix error in logistic regression.

FWFHP_clean = FWFHP_data.dropna(subset=["PHQ2_binary", "camp", "a_age", "gender", "a_phq2_score", "drug_use"])


print(f"Original rows: {len(FWFHP_data)}")
print(f"Clean rows: {len(FWFHP_clean)}")

print(FWFHP_data["camp"].value_counts())

# %%

# Logistic Regression without Camp 7 for odds ratios

import statsmodels.formula.api as smf

# Exclude camps with perfect separation (all PHQ2_binary = 0 or all = 1)
camp_outcome = FWFHP_clean.groupby("camp")["PHQ2_binary"].nunique()
valid_camps = camp_outcome[camp_outcome > 1].index
FWFHP_clean_no_camp7 = FWFHP_clean[FWFHP_clean["camp"].isin(valid_camps)]

# Likewise drop gender levels with only one outcome (e.g. a single Transgender
# record). Such singletons cause quasi-complete separation and stop the logistic
# model from converging (coefficient -> +/-inf with an enormous standard error).
gender_outcome = FWFHP_clean_no_camp7.groupby("gender")["PHQ2_binary"].nunique()
valid_genders = gender_outcome[gender_outcome > 1].index
FWFHP_clean_no_camp7 = FWFHP_clean_no_camp7[FWFHP_clean_no_camp7["gender"].isin(valid_genders)]

explor_fit = smf.logit("PHQ2_binary ~ C(camp) + a_age + gender + drug_use", data = FWFHP_clean_no_camp7).fit()

print(explor_fit.summary())          # coefficients (log-odds), SEs, p-values


# %%

# Gender Odd Ratios (gender is text so statsmodels names the param "gender[T.Male]")
gender_param = [p for p in explor_fit.params.index if p.startswith("gender")][0]
gender_odds = np.exp(explor_fit.params[gender_param])
ci_gender = np.exp(explor_fit.conf_int().loc[gender_param])

# Age Odd Ratios
age_odds = np.exp(explor_fit.params["a_age"])
ci_age = np.exp(explor_fit.conf_int().loc["a_age"])

# Drug Use Odds Ratio
drug_param = [p for p in explor_fit.params.index if p.startswith("drug_use")][0]
drug_odds  = np.exp(explor_fit.params[drug_param])
ci_drug    = np.exp(explor_fit.conf_int().loc[drug_param])

# Camp Odd Ratios
camp_odds = np.exp(explor_fit.params.filter(like="C(camp)"))
ci_camp = np.exp(explor_fit.conf_int().filter(like="C(camp)"))


# %%

# Forest Plot of Odds Ratios

# Build a combined dataframe of all predictors
rows = []

# Camp levels (reference = first alphabetical camp after exclusions)
ref_camp = sorted(valid_camps)[0]
ci_all = explor_fit.conf_int()
for label, or_val in camp_odds.items():
    camp_name = label.replace("C(camp)[T.", "").replace("]", "")
    rows.append({
        "Variable": f"{camp_name} vs. {ref_camp}",
        "OR": or_val,
        "Lower": np.exp(ci_all.loc[label].iloc[0]),
        "Upper": np.exp(ci_all.loc[label].iloc[1]),
    })

gender_label = gender_param.replace("gender[T.", "").replace("]", "")
drug_label   = drug_param.replace("drug_use[T.", "").replace("]", "")
rows.append({"Variable": "Age (per year)",           "OR": age_odds,   "Lower": ci_age[0],   "Upper": ci_age[1]})
rows.append({"Variable": f"Gender: {gender_label}",  "OR": gender_odds,"Lower": ci_gender[0],"Upper": ci_gender[1]})
rows.append({"Variable": f"Positive drug screen: {drug_label}",  "OR": drug_odds,  "Lower": ci_drug[0],  "Upper": ci_drug[1]})

forest_df = pd.DataFrame(rows)

# Plot
fig, ax = plt.subplots(figsize=(8, len(forest_df) * 0.6 + 1.5))

y_pos = range(len(forest_df) - 1, -1, -1)  # top-to-bottom order

for i, (_, row) in zip(y_pos, forest_df.iterrows()):
    ax.plot([row["Lower"], row["Upper"]], [i, i], color="steelblue", linewidth=2)  # CI line
    ax.plot(row["OR"], i, "o", color="steelblue", markersize=8, zorder=5)          # OR point
    # Single combined label pinned to right edge of axes, data y position
    label_transform = plt.matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.02, i, f'{row["OR"]:.2f} ({row["Lower"]:.2f}, {row["Upper"]:.2f})',
            ha="left", va="center", fontsize=8, transform=label_transform)

ax.axvline(x=1, color="black", linestyle="--", linewidth=1)  # null effect reference

ax.set_yticks(list(y_pos))
ax.set_yticklabels(forest_df["Variable"].tolist())
ax.set_xlabel("Odds Ratio (95% CI)", fontsize=12)
ax.set_title(f"Odds Ratios for PHQ-2 Score > 2\n(Logistic Regression, Reference: {ref_camp})", fontsize=13)
ax.set_xscale("log")  # log scale so CI whiskers appear symmetric
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
ax.grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("charts/forest_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
