# Graph Report - C:/Users/cruzm/FWFHP-Data  (2026-06-28)

## Corpus Check
- Corpus is ~19,344 words - fits in a single context window. You may not need a graph.

## Summary
- 89 nodes · 123 edges · 12 communities (9 shown, 3 thin omitted)
- Extraction: 75% EXTRACTED · 24% INFERRED · 1% AMBIGUOUS · INFERRED: 30 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_PHQ-2 Comparison Chart|PHQ-2 Comparison Chart]]
- [[_COMMUNITY_Diagnosis Count by PHQ-2 Group|Diagnosis Count by PHQ-2 Group]]
- [[_COMMUNITY_Project Scope and Design|Project Scope and Design]]
- [[_COMMUNITY_Logistic Regression Predictors|Logistic Regression Predictors]]
- [[_COMMUNITY_High PHQ-2 Diagnosis Probabilities|High PHQ-2 Diagnosis Probabilities]]
- [[_COMMUNITY_PHQ-2 Odds Ratios|PHQ-2 Odds Ratios]]
- [[_COMMUNITY_FWFHP vs NHANES Comparison|FWFHP vs NHANES Comparison]]
- [[_COMMUNITY_Co-occurrence Analysis Code|Co-occurrence Analysis Code]]
- [[_COMMUNITY_Slide Deck Components|Slide Deck Components]]
- [[_COMMUNITY_Graphify Config|Graphify Config]]

## God Nodes (most connected - your core abstractions)
1. `Chart: P(Diagnosis | PHQ-2 Score) High vs Low PHQ-2` - 16 edges
2. `Chart: Count of Diagnoses PHQ-2 > 2 vs PHQ-2 ≤ 2` - 16 edges
3. `Chart: P(Diagnosis | PHQ-2 Score > 2)` - 8 edges
4. `Odds Ratio Chart: PHQ-2 > 2 vs PHQ-2 ≤ 2` - 8 edges
5. `Forest Plot: Odds Ratios for PHQ-2 Score > 2` - 8 edges
6. `High PHQ-2 Group (PHQ-2 > 2, n=42)` - 7 edges
7. `PHQ-2 Screening Analysis Slide Deck — Night Camp Data, June 2026 (n=1,398)` - 6 edges
8. `PHQ-2 Depression Screening Score` - 6 edges
9. `PHQ-2 > 2 Group (n=42, Elevated Depressive Symptoms)` - 6 edges
10. `PHQ-2 ≤ 2 Group (n=1356, Low Depressive Symptoms)` - 6 edges

## Surprising Connections (you probably didn't know these)
- `NHANES Hispanic PHQ-2 Comparison Dataset (n=5,489, 12.4% positive screen)` --semantically_similar_to--> `Night Camp Two-Year Dataset (Demographic + PHQ-2 and Diagnosis Records)`  [INFERRED] [semantically similar]
  slides.html → read_june15.md
- `Odds Ratio Analytical Approach for PHQ-2 Co-occurrence (odds high / odds low group)` --semantically_similar_to--> `Co-occurrence Odds Ratios: Milaria 5.49, Dehydration 3.65, Elevated BP 2.90, Fatigue 2.18`  [INFERRED] [semantically similar]
  read_june15.md → slides.html
- `Slide Deck Creation Requirements: Findings from analysis.py and co_occurences.py` --conceptually_related_to--> `PHQ-2 Co-occurrence Analysis Project Scope`  [INFERRED]
  read_june17.md → read_june15.md
- `Co-occurrence Odds Ratios: Milaria 5.49, Dehydration 3.65, Elevated BP 2.90, Fatigue 2.18` --references--> `Diagnosis Co-occurrence Dataset (10 conditions: Dehydration, Dental Pain, Elevated BP, Fatigue, Headache, Insomnia, Milaria/Myalgia, Obesity, STI, Sunburn)`  [INFERRED]
  slides.html → read_june15.md
- `Slide Deck Creation Requirements: Findings from analysis.py and co_occurences.py` --references--> `PHQ-2 Screening Analysis Slide Deck — Night Camp Data, June 2026 (n=1,398)`  [EXTRACTED]
  read_june17.md → slides.html

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **PHQ-2 Co-occurrence Analysis Design (Inputs, Dataset, Method, Scope)** — read_june15_phq2_analysis_scope, read_june15_night_camp_data, read_june15_diagnosis_dataset, read_june15_odds_ratio_method [EXTRACTED 1.00]
- **Key Findings Presented in PHQ-2 Slide Deck** — slides_positive_screen_result, slides_co_occurrence_findings, slides_gender_predictor_finding, slides_nhanes_hispanic_comparison [INFERRED 0.95]

## Communities (12 total, 3 thin omitted)

### Community 0 - "PHQ-2 Comparison Chart"
Cohesion: 0.19
Nodes (17): Chart: P(Diagnosis | PHQ-2 Score) High vs Low PHQ-2, Conditional Probability of Diagnosis, Diagnosis: Dehydration, Diagnosis: Elevated BP, Diagnosis: Elevated Glucose, Diagnosis: Eye Pain, Diagnosis: Fatigue, Diagnosis: Headache (+9 more)

### Community 1 - "Diagnosis Count by PHQ-2 Group"
Cohesion: 0.20
Nodes (17): Chart: Count of Diagnoses PHQ-2 > 2 vs PHQ-2 ≤ 2, Diagnosis: Dehydration, Diagnosis: Elevated Blood Pressure, Diagnosis: Elevated Glucose, Diagnosis: Eye Pain, Diagnosis: Fatigue, Diagnosis: Headache, Diagnosis: Insomnia (+9 more)

### Community 2 - "Project Scope and Design"
Cohesion: 0.25
Nodes (11): Diagnosis Co-occurrence Dataset (10 conditions: Dehydration, Dental Pain, Elevated BP, Fatigue, Headache, Insomnia, Milaria/Myalgia, Obesity, STI, Sunburn), Night Camp Two-Year Dataset (Demographic + PHQ-2 and Diagnosis Records), Odds Ratio Analytical Approach for PHQ-2 Co-occurrence (odds high / odds low group), PHQ-2 Co-occurrence Analysis Project Scope, Slide Deck Creation Requirements: Findings from analysis.py and co_occurences.py, Co-occurrence Odds Ratios: Milaria 5.49, Dehydration 3.65, Elevated BP 2.90, Fatigue 2.18, Farm Worker Family Health Program (FWFHP), Gender as Significant Predictor of Positive PHQ-2 Screen (OR≈3.23, p=0.003) (+3 more)

### Community 3 - "Logistic Regression Predictors"
Cohesion: 0.36
Nodes (9): Age Predictor per Year (OR=0.98, 95% CI: 0.94-1.02, Non-Significant), Camp 1 (Reference Group), Camp-Level Comparisons (Camps 2,3,5,6,8,9 vs Camp 1), Forest Plot: Odds Ratios for PHQ-2 Score > 2, Gender Predictor (OR=3.23, 95% CI: 1.48-7.05, Significant), Logistic Regression (Statistical Method), Null Finding: No Significant Camp Differences in PHQ-2 Risk, Odds Ratio with 95% CI (Effect Measure) (+1 more)

### Community 4 - "High PHQ-2 Diagnosis Probabilities"
Cohesion: 0.36
Nodes (9): Chart: P(Diagnosis | PHQ-2 Score > 2), Conditional Probability of Diagnosis, Dehydration Diagnosis (~2.4% conditional probability), Elevated BP Diagnosis (~7.1% conditional probability), Fatigue Diagnosis (~2.4% conditional probability), Milaria Diagnosis (~2.4% conditional probability), Near-Zero Diagnoses (Eye Pain, Headache, Insomnia, Myalgia, Obesity, STIs, Sunburn, Elevated Glucose), PHQ-2 Score > 2 (High PHQ-2 Threshold) (+1 more)

### Community 5 - "PHQ-2 Odds Ratios"
Cohesion: 0.39
Nodes (9): Odds Ratio Chart: PHQ-2 > 2 vs PHQ-2 ≤ 2, Dehydration Diagnosis (OR ~3.6, second highest association with elevated PHQ-2), Elevated Blood Pressure Diagnosis (OR ~2.9), Fatigue Diagnosis (OR ~2.2), Milaria Diagnosis (OR ~5.5, highest association with elevated PHQ-2), Diagnoses with OR near zero or undefined: Elevated Glucose, Eye Pain, Headache, Insomnia, Myalgia, STIs, Sunburn, Obesity Diagnosis (OR undefined — 0 cases in both PHQ-2 groups), Odds Ratio Statistical Measure (+1 more)

### Community 6 - "FWFHP vs NHANES Comparison"
Cohesion: 0.36
Nodes (9): Pie Chart: PHQ-2 Score Distribution (FWFHP vs NHANES Hispanic), Insight: FWFHP has markedly lower PHQ-2 positive rate (2.8%) than NHANES Hispanic (12.4%), suggesting underreporting or healthier sample, FWFHP PHQ-2 > 2: 2.8%, FWFHP PHQ-2 <= 2: 97.2%, FWFHP Sample (n=1501), NHANES Hispanic Sample (n=5489), NHANES Hispanic PHQ-2 > 2: 12.4%, NHANES Hispanic PHQ-2 <= 2: 87.6% (+1 more)

## Ambiguous Edges - Review These
- `PHQ-2 Depression Screening Score` → `Obesity Diagnosis (OR undefined — 0 cases in both PHQ-2 groups)`  [AMBIGUOUS]
  charts/chart3_odds_ratios.png · relation: conceptually_related_to

## Knowledge Gaps
- **25 isolated node(s):** `Farm Worker Family Health Program (FWFHP)`, `Gender as Significant Predictor of Positive PHQ-2 Screen (OR≈3.23, p=0.003)`, `PHQ-2 Positive Screen Result: 42 patients (3.0%) vs 1,356 negative (97.0%)`, `Deck (JavaScript Slide Controller Class)`, `Editor (JavaScript Inline Editor Class with contenteditable toggle)` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `PHQ-2 Depression Screening Score` and `Obesity Diagnosis (OR undefined — 0 cases in both PHQ-2 groups)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What connects `Odds ratio with a note when undefined.     Returns (OR, note) where note is ''`, `Graphify Knowledge Graph Workflow Rules`, `Farm Worker Family Health Program (FWFHP)` to the rest of the system?**
  _27 weakly-connected nodes found - possible documentation gaps or missing edges._