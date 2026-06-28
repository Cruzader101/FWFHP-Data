# %%

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt

NHANES_data = pd.read_csv("./NHANES/df_Hispanic_NHANES.csv")
FWFHP_data = pd.read_csv("./Night Camp/FarmWorkerFamilyHeal-PHQ2WithDiagnosis_DATA_2026-06-15_0914.csv")

# %%

#Adding binary column for PHQ-2 score greater than 2 for both datasets

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

# Odds Ratio

odds_FWFHP = len(FWFHP_yes) / len(FWFHP_no)
odds_NHANES = len(NHANES_yes) / len(NHANES_no)

print("Odds FWFHP:", odds_FWFHP)
print("Odds NHANES:", odds_NHANES)

odds_ratio = odds_FWFHP / odds_NHANES

print("Odds Ratio:", odds_ratio)

# %%

# Significance Testing

# %%

# Fixing missing data, causing singular matrix error in logistic regression.

FWFHP_clean = FWFHP_data.dropna(subset=["PHQ2_binary", "camp", "a_age", "gender", "a_phq2_score"])

print(f"Original rows: {len(FWFHP_data)}")
print(f"Clean rows: {len(FWFHP_clean)}")

print(FWFHP_data["camp"].value_counts())

# %%

# Logistic Regression without Camp 7 for odds ratios

import statsmodels.formula.api as smf

FWFHP_clean_no_camp7 = FWFHP_clean[FWFHP_clean["camp"] != 7]

explor_fit = smf.logit("PHQ2_binary ~ C(camp) + a_age + gender", data = FWFHP_clean_no_camp7).fit()

print(explor_fit.summary())          # coefficients (log-odds), SEs, p-values


# %%

# Gender Odd Ratios
gender_odds = np.exp(explor_fit.params["gender"]) # 
ci_gender = np.exp(explor_fit.conf_int().loc["gender"]) # confidence interval

# Age Odd Ratios
age_odds = np.exp(explor_fit.params["a_age"])
ci_age = np.exp(explor_fit.conf_int().loc["a_age"])

# Camp Odd Ratios 
camp_odds = np.exp(explor_fit.params.filter(like="C(camp)"))
ci_camp = np.exp(explor_fit.conf_int().filter(like="C(camp)"))


# %%

# Forest Plot of Odds Ratios

# Build a combined dataframe of all predictors
rows = []

# Camp levels (reference = camp 1, excluded from params)
ci_all = explor_fit.conf_int()  # full CI table; columns are positional (iloc), not named
for label, or_val in camp_odds.items():
    camp_num = label.replace("C(camp)[T.", "").replace("]", "")
    rows.append({
        "Variable": f"Camp {camp_num} vs. Camp 1",
        "OR": or_val,
        "Lower": np.exp(ci_all.loc[label].iloc[0]),
        "Upper": np.exp(ci_all.loc[label].iloc[1]),
    })

rows.append({"Variable": "Age (per year)", "OR": age_odds, "Lower": ci_age[0], "Upper": ci_age[1]})
rows.append({"Variable": "Gender", "OR": gender_odds, "Lower": ci_gender[0], "Upper": ci_gender[1]})

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
ax.set_title("Odds Ratios for PHQ-2 Score > 2\n(Logistic Regression, Reference: Camp 1)", fontsize=13)
ax.set_xscale("log")  # log scale so CI whiskers appear symmetric
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
ax.grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("forest_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
