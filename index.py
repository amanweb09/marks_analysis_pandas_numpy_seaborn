import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

marksheet = pd.read_csv("marksheet.csv")

# **** WRANGLING ****

marksheet = marksheet.drop("Unnamed: 0", axis=1)  # dropping unnamed column

nulls = marksheet.isnull().sum()  # checking for null values

marksheet = marksheet.dropna()  # drop null values


# **** DATA TRANSFORMATION ****

marksheet["WklyStudyHours"] = marksheet["WklyStudyHours"].str.replace("05-Oct", "5-10")


# **** BASIC INSIGHTS ****
desc = marksheet.describe()
"""
** Insights:
    1. No one scored 0 in reading and writing, while some children scored 0 in maths
"""


# **** GRAPHS ****

# 1. Gender Distribution
gender_plot = sns.countplot(
    data=marksheet,
    x="Gender",
)
gender_plot.bar_label(gender_plot.containers[0])
plt.show()

"""
** Insights
  2. Number of females in the data is more than males
"""

# 2. Impact of parent education on marks
grouped_by_pEdu = marksheet.groupby("ParentEduc")
impact = grouped_by_pEdu.agg(
    {
        "MathScore": "mean",
        "ReadingScore": "mean",
        "WritingScore": "mean",
    }
)
sns.heatmap(
    impact,
    annot=True,
)
plt.show()
"""
** INSIGHTS
  3. Parent's education has a great impact on the scores of their children
  Children whose parents have a master's degree tend to perform much better than 
  those who did just high school
"""

# 3. Impact of parent's marital status
grouped_by_marital_status = marksheet.groupby("ParentMaritalStatus")
impact_1 = grouped_by_marital_status.agg(
    {
        "MathScore": "mean",
        "ReadingScore": "mean",
        "WritingScore": "mean",
    }
)
sns.heatmap(
    impact_1,
    annot=True,
)
plt.show()
"""
** INSIGHTS
4. Parent's marital status has negligible impact on their children's marks
"""

# 4. Impact of weekly study hours
grouped_by_study_hours = marksheet.groupby("WklyStudyHours")
impact_2 = grouped_by_study_hours.agg(
    {
        "MathScore": "mean",
        "ReadingScore": "mean",
        "WritingScore": "mean",
    }
)
sns.heatmap(impact_2, annot=True)
plt.show()
"""
** INSIGHTS
5. Surprisingly, study hours show negligible impact on the average scores.
"""

# 5. Distribution of ethnic groups
counts = {}
eth_groups = marksheet["EthnicGroup"].unique()
for group in eth_groups:
    counts[group] = marksheet.loc[marksheet["EthnicGroup"] == group][
        "EthnicGroup"
    ].count()

plt.pie(
    counts.values(),
    labels=counts.keys(),
)
plt.show()
"""
** INSIGHTS
  6. Majority of the students are from group C
"""