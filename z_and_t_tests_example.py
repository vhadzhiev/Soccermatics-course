import pandas as pd
import json
import warnings

from statsmodels.stats.weightstats import ztest
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

#open events
train = pd.DataFrame()
path = '.\data\wyscout\data\events\events_Italy.json'
with open(path) as f:
    data = json.load(f)
train = pd.concat([train, pd.DataFrame(data)])

#open team data
path = '.\data\wyscout\data\\teams.json'
with open(path) as f:
    teams = json.load(f)
teams_df = pd.DataFrame(teams)
teams_df = teams_df.rename(columns={"wyId": "teamId"})

#get corners
corners = train.loc[train["subEventName"] == "Corner"]

#count corners by team
corners_by_team = corners.groupby(['teamId']).size().reset_index(name='counts')
summary = corners_by_team.merge(teams_df[["name", "teamId"]], how = "left", on = ["teamId"])

#count corners by team by game
corners_by_game = corners.groupby(['teamId', "matchId"]).size().reset_index(name='counts')
summary2 = corners_by_game.merge(teams_df[["name", "teamId"]], how = "left", on = ["teamId"])

#get Sampdoria corners
samp_corners = summary2.loc[summary2["name"] == 'Sampdoria']["counts"]
t, pvalue = ztest(samp_corners,  value=7)

if pvalue < 0.05:
    print("P-value amounts to", pvalue, " - We reject null hypothesis - Sampdoria do not take 7 corners per game")
else:
    print("P-value amounts to", pvalue, " - We do not reject null hypothesis - Sampdoria take 7 corners per game")

t, pvalue = ztest(samp_corners,  value=6, alternative = "larger")
if pvalue < 0.05:
    print("P-value amounts to", pvalue, " - We reject null hypothesis - Sampdoria take more than 6 corners per game")
else:
    print("P-value amounts to", pvalue, " - We do not reject null hypothesis - Sampdoria do not take 6 more corners per game")
    
    
mean = summary["counts"].mean()
std = summary["counts"].std()

samp_corners = summary.loc[summary["name"] == "Sampdoria"]["counts"].iloc[0]
t, pvalue = ttest_1samp(summary["counts"], samp_corners)

if pvalue < 0.05:
    print("P-value amounts to", pvalue, " - We reject null hypothesis - Sampdoria do not take average number of corners more than league average")
else:
    print("P-value amounts to", pvalue, " - We do not reject null hypothesis - Sampdoria take average number of corners less than league average")
    

samp_corners = summary2.loc[summary2["name"] == 'Sampdoria']["counts"]
genoa_corners = summary2.loc[summary2["name"] == 'Genoa']["counts"]

t, pvalue  = ttest_ind(a=samp_corners, b=genoa_corners, equal_var=True)

if pvalue < 0.05:
    print("P-value amounts to", pvalue, " - We reject null hypothesis - Sampdoria took different number of corners per game than Genoa")
else:
    print("P-value amounts to", pvalue, " - We do not reject null hypothesis - Sampdoria took the same number of corners per game as Genoa")
    

samp_corners = summary2.loc[summary2["name"] == 'Sampdoria']["counts"]
genoa_corners = summary2.loc[summary2["name"] == 'Genoa']["counts"]

t, pvalue  = ttest_ind(a=samp_corners, b=genoa_corners, equal_var=True, alternative = "greater")

if pvalue < 0.05:
    print("P-value amounts to", pvalue, " - We reject null hypothesis - Sampdoria took more corners per game than Genoa")
else:
    print("P-value amounts to", pvalue, " - We do not reject null hypothesis - Sampdoria did not  take the more corners per game than Genoa")