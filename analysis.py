# Python Code for Obtaining the Analysis on the Survey Results
# First Install the following dependencies using the commands listed for them.
# 1.	pandas
# Pip install pandas
# 2.	numpy
# Pip install numpy
# 3.	matplotlib
# Pip install matplotlib
# 4.	scipy
# Pip install scipy
# 5.	Statsmodels
# Pip install Statsmodels
# 6.	openpyxl
# Pip install openpyxl

# Note: The final results would be saved in the output folder, within the analysis folder.
# Following is the code for Analysis.py file.
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# End-to-end analysis for Chapters 5–7:
# - Loads Survey_Responses_100.xlsx (local Excel)
# - Computes indices, reliability, descriptives, correlations
# - Runs OLS regression, ANOVA, Tukey HSD
# - Generates Tables 5.1–5.5 (CSV) and Figures 5.1–5.6 (PNG)
# - Saves everything under ./output

# Usage:
#     pip install -r requirements.txt
#     python run_analysis.py
# """

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# Config
# ---------------------------
INPUT_XLSX = "Survey_Responses_100.xlsx"   # keep in same folder
OUTPUT_DIR = "output"

# Dissertation style
matplotlib.rcParams["font.family"] = "Times New Roman"

PALETTE = ["#0077b6", "#0096c7"]  # blue/teal

LIKERT_COLS = [
    "Q4_StrategicPlanDefined","Q5_ObjectivesCommunicated","Q6_FrameworksUsed","Q7_StrategyReviewed",
    "Q8_StrategyDrivesPerformance","Q9_StrategyToOutcomes","Q10_InitiativesAchieveObjectives",
    "Q11_ImplementationChallenges","Q12_AdaptationAgility","Q13_UsesBestPracticesVsPeers","Q14_ImprovementWouldHelp"
]
PRACTICE_ITEMS = ["Q4_StrategicPlanDefined","Q5_ObjectivesCommunicated","Q6_FrameworksUsed","Q7_StrategyReviewed"]
OUTCOME_ITEMS  = ["Q8_StrategyDrivesPerformance","Q9_StrategyToOutcomes","Q10_InitiativesAchieveObjectives"]

# ---------------------------
# Helpers
# ---------------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def cronbach_alpha(df_items: pd.DataFrame) -> float:
    x = df_items.dropna()
    k = x.shape[1]
    if k < 2:
        return np.nan
    var_sum = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k/(k-1)) * (1 - var_sum/total_var)

def save_csv(df: pd.DataFrame, filename: str):
    outpath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(outpath, index=False)
    return outpath

def save_fig(fig, filename: str):
    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath

def load_excel_any_sheet(path: str) -> pd.DataFrame:
    # Try sheet named "Responses"; if not, load the first sheet.
    try:
        return pd.read_excel(path, sheet_name="Responses")
    except Exception:
        return pd.read_excel(path)

# ---------------------------
# Main
# ---------------------------
def main():
    if not os.path.exists(INPUT_XLSX):
        sys.exit(f"ERROR: '{INPUT_XLSX}' not found in current folder.")

    ensure_outdir(OUTPUT_DIR)

    # ---- Load
    df = load_excel_any_sheet(INPUT_XLSX)

    # ---- Sanitize data types
    for c in LIKERT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            sys.exit(f"ERROR: Missing expected column '{c}' in dataset.")

    # ---- Indices
    df["Practice_Index"] = df[PRACTICE_ITEMS].mean(axis=1)
    df["Outcome_Index"]  = df[OUTCOME_ITEMS].mean(axis=1)

    # ---- Reliability
    alpha_practice = cronbach_alpha(df[PRACTICE_ITEMS])
    alpha_outcome  = cronbach_alpha(df[OUTCOME_ITEMS])

    # ---- Normality (optional)
    shapiro_practice = stats.shapiro(df["Practice_Index"].dropna())
    shapiro_outcome  = stats.shapiro(df["Outcome_Index"].dropna())

    # ---- Table 5.1 (Descriptives)
    rows = []
    for c in LIKERT_COLS + ["Practice_Index","Outcome_Index"]:
        s = df[c].dropna()
        rows.append([
            c,
            round(s.mean(), 2),
            round(s.std(ddof=1), 2),
            int(s.min()) if len(s) else np.nan,
            int(s.max()) if len(s) else np.nan,
            int(s.shape[0])
        ])
    desc_df = pd.DataFrame(rows, columns=["Variable","Mean","SD","Min","Max","N"])
    save_csv(desc_df, "Table_5_1_Descriptives.csv")

    # ---- Table 5.2 (Correlations)
    series_map = {
        "Practice_Index": df["Practice_Index"],
        "Outcome_Index": df["Outcome_Index"],
        "Implementation_Challenges": df["Q11_ImplementationChallenges"],
        "Adaptation_Agility": df["Q12_AdaptationAgility"],
        "Best_Practices_vs_Peers": df["Q13_UsesBestPracticesVsPeers"]
    }
    order = list(series_map.keys())
    corr_display = pd.DataFrame(index=order, columns=order, dtype=object)
    for i in order:
        for j in order:
            if i == j:
                corr_display.loc[i, j] = "—"
            else:
                a, b = series_map[i], series_map[j]
                m = a.notna() & b.notna()
                r, p = stats.pearsonr(a[m], b[m])
                corr_display.loc[i, j] = f"{r:.2f} (p={p:.3f})"
    save_csv(corr_display.reset_index().rename(columns={"index":"Variable"}), "Table_5_2_Correlations.csv")

    # ---- Table 5.3 (Regression with moderation)
    df["Practice_x_Agility"] = df["Practice_Index"] * df["Q12_AdaptationAgility"]
    X = df[["Practice_Index","Q11_ImplementationChallenges","Q12_AdaptationAgility","Practice_x_Agility"]].copy()
    X = sm.add_constant(X)
    y = df["Outcome_Index"]
    reg = sm.OLS(y, X, missing="drop").fit()
    reg_tbl = reg.summary2().tables[1].round(3).reset_index().rename(columns={"index":"Predictor"})
    save_csv(reg_tbl, "Table_5_3_Regression.csv")

    # ---- Table 5.4 (ANOVA by Role)
    df["Role"] = df["Role"].astype(str)
    anova_model = smf.ols("Outcome_Index ~ C(Role)", data=df).fit()
    anova_tbl = anova_lm(anova_model, typ=2).round(3).reset_index().rename(columns={"index":"Source"})
    save_csv(anova_tbl, "Table_5_4_ANOVA.csv")

    # ---- Table 5.5 (Tukey HSD)
    tukey = pairwise_tukeyhsd(endog=df["Outcome_Index"], groups=df["Role"], alpha=0.05)
    tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0]).round(3)
    save_csv(tukey_df, "Table_5_5_Tukey.csv")

    # ---------------------------
    # Figures (PNG)
    # ---------------------------

    # Figure 5.1 – Role pie
    role_counts = df["Role"].value_counts()
    fig = plt.figure(figsize=(6,6))
    plt.pie(role_counts, labels=role_counts.index, autopct='%1.1f%%', startangle=90,
            colors=[PALETTE[i % 2] for i in range(len(role_counts))])
    plt.title("Figure 5.1 – Distribution of Respondents by Role", fontsize=12)
    save_fig(fig, "Figure_5_1_Role.png")

    # Figure 5.2 – Industry pie
    ind_counts = df["Industry"].astype(str).value_counts()
    fig = plt.figure(figsize=(6,6))
    plt.pie(ind_counts, labels=ind_counts.index, autopct='%1.1f%%', startangle=90,
            colors=[PALETTE[i % 2] for i in range(len(ind_counts))])
    plt.title("Figure 5.2 – Distribution of Respondents by Industry", fontsize=12)
    save_fig(fig, "Figure_5_2_Industry.png")

    # Figure 5.3 – Means ± SD (Q4–Q14)
    means = df[LIKERT_COLS].mean()
    sds   = df[LIKERT_COLS].std(ddof=1)
    labels = [
        "Q4 Strategic Plan Defined", "Q5 Objectives Communicated", "Q6 Frameworks Used",
        "Q7 Strategy Reviewed", "Q8 Strategy Drives Performance", "Q9 Strategy to Outcomes",
        "Q10 Initiatives Achieve Objectives", "Q11 Implementation Challenges", "Q12 Adaptation Agility",
        "Q13 Best Practices vs Peers", "Q14 Improvement Would Help"
    ]
    fig = plt.figure(figsize=(12,6))
    x = np.arange(len(labels))
    plt.bar(x, means.values, yerr=sds.values, capsize=4,
            color=[PALETTE[i % 2] for i in range(len(labels))],
            edgecolor='black', linewidth=0.3)
    plt.xticks(x, labels, rotation=35, ha='right')
    plt.ylabel("Mean Response (1–5 Likert Scale)")
    plt.ylim(0, 5)
    plt.title("Figure 5.3 – Mean (±SD) of Strategic Management Survey Items", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "Figure_5_3_Means.png")

    # Figure 5.4 – Practice vs Outcome (scatter + fit)
    xvals = df["Practice_Index"].values
    yvals = df["Outcome_Index"].values
    coef = np.polyfit(xvals, yvals, 1)
    poly1d_fn = np.poly1d(coef)
    fig = plt.figure(figsize=(7,6))
    plt.scatter(xvals, yvals, color=PALETTE[0], alpha=0.75, edgecolor="black", linewidth=0.3, label="Respondents")
    # Fit line
    xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 100)
    plt.plot(xs, poly1d_fn(xs), color=PALETTE[1], linewidth=2.2, label="Best fit line")
    plt.xlabel("Practice Index (1–5 Likert)")
    plt.ylabel("Outcome Index (1–5 Likert)")
    plt.title("Figure 5.4 – Practice Index vs Outcome Index", fontsize=13)
    plt.xlim(1, 5); plt.ylim(1, 5)
    plt.legend(frameon=False)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    save_fig(fig, "Figure_5_4_Practice_vs_Outcome.png")

    # Figure 5.5 – Outcome by Role (boxplot)
    order_roles = role_counts.index.tolist()
    data_groups = [df.loc[df["Role"]==r, "Outcome_Index"].dropna().values for r in order_roles]
    fig = plt.figure(figsize=(10,5))
    plt.boxplot(data_groups, labels=order_roles, showmeans=True)
    plt.ylabel("Outcome Index")
    plt.title("Figure 5.5 – Outcome Index by Role", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "Figure_5_5_Outcome_by_Role.png")

    # Figure 5.6 – Moderation (Practice × Agility simple slopes)
    ag_mean = df["Q12_AdaptationAgility"].mean()
    ag_sd   = df["Q12_AdaptationAgility"].std(ddof=1)
    low_ag  = ag_mean - ag_sd
    high_ag = ag_mean + ag_sd
    bpar = reg.params
    px = np.linspace(np.nanmin(df["Practice_Index"]), np.nanmax(df["Practice_Index"]), 50)

    def predict_out(prac, agil, ch_mean):
        return (bpar["const"]
                + bpar["Practice_Index"]*prac
                + bpar["Q11_ImplementationChallenges"]*ch_mean
                + bpar["Q12_AdaptationAgility"]*agil
                + bpar["Practice_x_Agility"]*(prac*agil))

    ch_mean = df["Q11_ImplementationChallenges"].mean()
    y_low  = np.array([predict_out(p, low_ag,  ch_mean) for p in px])
    y_high = np.array([predict_out(p, high_ag, ch_mean) for p in px])

    fig = plt.figure(figsize=(7,5))
    plt.plot(px, y_low,  label=f"Low Agility (~{low_ag:.2f})",  color=PALETTE[0], linewidth=2.0)
    plt.plot(px, y_high, label=f"High Agility (~{high_ag:.2f})", color=PALETTE[1], linewidth=2.0)
    plt.xlabel("Practice Index")
    plt.ylabel("Predicted Outcome Index")
    plt.title("Figure 5.6 – Moderation: Practice × Agility", fontsize=13)
    plt.legend(frameon=False)
    plt.tight_layout()
    save_fig(fig, "Figure_5_6_Moderation.png")

    # ---- Console summary
    print("\n=== SUMMARY ===")
    print(f"Cronbach α (Practice): {alpha_practice:.2f}")
    print(f"Cronbach α (Outcome):  {alpha_outcome:.2f}")
    print(f"Shapiro–Wilk (Practice): W={shapiro_practice.statistic:.3f}, p={shapiro_practice.pvalue:.3f}")
    print(f"Shapiro–Wilk (Outcome):  W={shapiro_outcome.statistic:.3f}, p={shapiro_outcome.pvalue:.3f}")
    print(f"Regression R²: {reg.rsquared:.2f}, Adjusted: {reg.rsquared_adj:.2f}")
    print("\nAll tables and figures saved under './output'. Done.")

if __name__ == "__main__":
    main()

