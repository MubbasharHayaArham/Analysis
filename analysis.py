import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# Ensure output folder
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# Load Data
xlsx = "Survey_Responses_100.xlsx"
try:
    df = pd.read_excel(xlsx, sheet_name="Responses")
except:
    df = pd.read_excel(xlsx)

# Convert Likert columns to numeric
likert_cols = [
    "Q4_StrategicPlanDefined","Q5_ObjectivesCommunicated","Q6_FrameworksUsed","Q7_StrategyReviewed",
    "Q8_StrategyDrivesPerformance","Q9_StrategyToOutcomes","Q10_InitiativesAchieveObjectives",
    "Q11_ImplementationChallenges","Q12_AdaptationAgility","Q13_UsesBestPracticesVsPeers","Q14_ImprovementWouldHelp"
]
for c in likert_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Create indices
practice_items = ["Q4_StrategicPlanDefined","Q5_ObjectivesCommunicated","Q6_FrameworksUsed","Q7_StrategyReviewed"]
outcome_items  = ["Q8_StrategyDrivesPerformance","Q9_StrategyToOutcomes","Q10_InitiativesAchieveObjectives"]
df["Practice_Index"] = df[practice_items].mean(axis=1)
df["Outcome_Index"]  = df[outcome_items].mean(axis=1)

# Cronbach’s Alpha
def cronbach_alpha(items):
    x = items.dropna()
    k = x.shape[1]
    var_sum = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    return (k/(k-1)) * (1 - var_sum/total_var)

alpha_practice = cronbach_alpha(df[practice_items])
alpha_outcome  = cronbach_alpha(df[outcome_items])

# Table 5.1 – Descriptive Statistics
desc = []
for c in likert_cols + ["Practice_Index","Outcome_Index"]:
    s = df[c].dropna()
    desc.append([c, round(s.mean(),2), round(s.std(ddof=1),2),
                 int(s.min()), int(s.max()), len(s)])
pd.DataFrame(desc, columns=["Variable","Mean","SD","Min","Max","N"]).to_csv(f"{OUT}/Table_5_1_Descriptives.csv", index=False)

# Table 5.2 – Correlation Matrix
series = {
    "Practice_Index": df["Practice_Index"],
    "Outcome_Index": df["Outcome_Index"],
    "Challenges": df["Q11_ImplementationChallenges"],
    "Agility": df["Q12_AdaptationAgility"],
    "Best_Practices": df["Q13_UsesBestPracticesVsPeers"]
}
order = list(series.keys())
corr = pd.DataFrame(index=order, columns=order, dtype=object)
for i in order:
    for j in order:
        if i == j:
            corr.loc[i,j] = "—"
        else:
            m = series[i].notna() & series[j].notna()
            r,p = stats.pearsonr(series[i][m], series[j][m])
            corr.loc[i,j] = f"{r:.2f} (p={p:.3f})"
corr.to_csv(f"{OUT}/Table_5_2_Correlations.csv")

# Table 5.3 – Moderated Regression
df["Practice_x_Agility"] = df["Practice_Index"] * df["Q12_AdaptationAgility"]
X = df[["Practice_Index","Q11_ImplementationChallenges","Q12_AdaptationAgility","Practice_x_Agility"]]
X = sm.add_constant(X)
reg = sm.OLS(df["Outcome_Index"], X, missing="drop").fit()
reg.summary2().tables[1].to_csv(f"{OUT}/Table_5_3_Regression.csv")

# Table 5.4 – ANOVA by Role
df["Role"] = df["Role"].astype(str)
anova_result = anova_lm(smf.ols("Outcome_Index ~ C(Role)", df).fit(), typ=2)
anova_result.to_csv(f"{OUT}/Table_5_4_ANOVA.csv")

# Table 5.5 – Tukey HSD Post-Hoc
tukey = pairwise_tukeyhsd(df["Outcome_Index"], df["Role"])
pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])\
  .to_csv(f"{OUT}/Table_5_5_Tukey.csv", index=False)

# Figures
plt.pie(df["Role"].value_counts(), labels=df["Role"].value_counts().index, autopct="%1.1f%%")
plt.title("Distribution by Role")
plt.savefig(f"{OUT}/Figure_5_1_Role.png", dpi=200, bbox_inches="tight"); plt.close()

plt.pie(df["Industry"].value_counts(), labels=df["Industry"].value_counts().index, autopct="%1.1f%%")
plt.title("Distribution by Industry")
plt.savefig(f"{OUT}/Figure_5_2_Industry.png", dpi=200, bbox_inches="tight"); plt.close()

print("\n All tables + figures saved in /outputs folder")
