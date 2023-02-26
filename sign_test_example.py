# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:25:03 2023

@author: USER
"""

import pandas as pd
import json
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


train = pd.DataFrame()
path = '.\data\wyscout\data\events\events_Italy.json'
with open(path) as f:
    data = json.load(f)
train = pd.concat([train, pd.DataFrame(data)])

path = '.\data\wyscout\data\players.json'
with open(path) as f:
    players = json.load(f)
player_df = pd.DataFrame(players)

#take shots only
shots = train.loc[train['subEventName'] == 'Shot']
#look for Quagliarella's id
quagliarella_id = player_df.loc[player_df["lastName"] == "Quagliarella"]["wyId"].iloc[0]
#get Quagliarella's shot
quagliarella_shots = shots.loc[shots["playerId"] == quagliarella_id]

#left leg shots
lefty_shots = quagliarella_shots.loc[quagliarella_shots.apply (lambda x:{'id':401} in x.tags, axis = 1)]
#right leg shots
righty_shots = quagliarella_shots.loc[quagliarella_shots.apply (lambda x:{'id':402} in x.tags, axis = 1)]

#create list with ones for left foot shots and -1 for right foot shots
shots_list = [1] * len(lefty_shots)
shots_list.extend([-1] * len(righty_shots))

from statsmodels.stats.descriptivestats import sign_test
test = sign_test(shots_list, mu0 = 0)
pvalue = test[1]

if pvalue < 0.05:
    print("P-value amounts to", str(pvalue)[:5], " - We reject null hypothesis - Fabio Quagliarella is not ambidextrous")
else:
    print("P-value amounts to", str(pvalue)[:5], " - We do not reject null hypothesis - Fabio Quagliarella is ambidextrous")