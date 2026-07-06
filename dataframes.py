# %%

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import pyreadstat as pr

# Loading PHQ-2 data, total score over 2 and percentage

df_PHQ2 = pd.read_csv("./Night Camp Data/FarmWorkerFamilyHeal-DemographicDataWithP_DATA_LABELS_2026-06-28_1652.csv")
df_PHQ2 = df_PHQ2.rename(columns={"Record ID": "record_id", "PHQ-2 Score": "a_phq2_score"})

# Restrict to respondents with a valid PHQ-2 score so the denominator matches the
# other scripts (rows with a missing score are neither positive nor "<= 2").
df_PHQ2 = df_PHQ2.dropna(subset=["a_phq2_score"])

count_FWFHP = (df_PHQ2["a_phq2_score"] > 2).sum()

total_length_FWFHP = len(df_PHQ2)

percentage_FWFHP = count_FWFHP / total_length_FWFHP * 100

print(count_FWFHP)
print(percentage_FWFHP)

# %%

# Labeling Diagnosis in FWFHP dataset




# %%

# Loading NHANES data 2023
# DPQ_L.xpt is an XPORT file, not a sas7bdat file.
df_DPQ_NHANES, meta = pr.read_xport("DPQ_L.xpt")

# DPQ010/DPQ020 use 7 = refused, 9 = don't know as non-response codes.
# A valid PHQ-2 needs BOTH items answered; any 7/9 on either item invalidates the row.
mask010 = df_DPQ_NHANES["DPQ010"].isin([7, 9])
mask020 = df_DPQ_NHANES["DPQ020"].isin([7, 9])

invalid = mask010 | mask020  # either item refused/don't-know

print("Rows with a refused/don't-know item:", int(invalid.sum()))
print(df_DPQ_NHANES.loc[invalid, ["DPQ010", "DPQ020"]].head(20))

# %%

# Total score for DPQ010 + DPQ020, only for rows where both items are valid answers.
# (Summing a 7/9 code would fabricate a high score, e.g. 9 + 9 = 18.)

df_DPQ_NHANES["phq2_score"] = np.where(
    ~invalid,
    df_DPQ_NHANES["DPQ010"] + df_DPQ_NHANES["DPQ020"],
    pd.NA
)

# %%

# Counting how many have a PHQ-2 for NHANES score greater than 2

count_NHANES = (df_DPQ_NHANES["phq2_score"] > 2).sum()

total_valid_NHANES = df_DPQ_NHANES["phq2_score"].notna().sum()

percentage_NHANES = count_NHANES / total_valid_NHANES * 100

print(count_NHANES)
print(percentage_NHANES)

# %%

# Creating pie charts to show percentages

# Data for pie charts
labels_pie = ['PHQ-2 > 2', 'PHQ-2 ≤ 2']
fwfhp_data = [count_FWFHP, total_length_FWFHP - count_FWFHP]
nhanes_data = [count_NHANES, total_valid_NHANES - count_NHANES]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# FWFHP pie chart
ax1.pie(fwfhp_data, labels=labels_pie, autopct='%1.1f%%', startangle=90)
ax1.set_title(f'FWFHP (n={total_length_FWFHP})')

# NHANES pie chart
ax2.pie(nhanes_data, labels=labels_pie, autopct='%1.1f%%', startangle=90)
ax2.set_title(f'NHANES (n={total_valid_NHANES})')

plt.tight_layout()
plt.show()



# %%

# Labeling by Ethnicity in NHANES,

df_Demo_NHANES, meta = pr.read_xport("DEMO_L.xpt", encoding="latin1")


ethinicity_labels = {1: 'Mexican American', 
2: 'Other Hispanic', 3: 'Non-Hispanic White', 
4: 'Non-Hispanic Black', 6: 'Non-Hispanic Asian', 
7: 'Other Race - Including Multi-Ravial',} 

df_Demo_NHANES["race_eth"] = df_Demo_NHANES["RIDRETH3"].map(ethinicity_labels).fillna("Missing")


# %%

# merging the two NHANES datasets on SEQN

demo_col = df_Demo_NHANES[['SEQN', 'RIDRETH3', 'race_eth']]
DPQ_col = df_DPQ_NHANES[['SEQN', 'DPQ010', 'DPQ020', 'phq2_score']]

df_merged_NHANES = pd.merge(demo_col, DPQ_col, on='SEQN', how='inner')


# %%

# Filtering by Mexican American and Other-Hispanic and then exporting data

df_Hispanic_NHANES = df_merged_NHANES[df_merged_NHANES['RIDRETH3'].isin([1, 2])]

df_Hispanic_NHANES.to_csv("./NHANES/df_Hispanic_NHANES.csv", index=False)

# %%
