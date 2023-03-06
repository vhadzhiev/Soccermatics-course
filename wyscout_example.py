import pandas as pd
import json

path = '.\data\wyscout\data\competitions.json'
with open(path) as f:
    data = json.load(f)
    
df_competitions = pd.DataFrame(data)
df_competitions.info()

path = '.\data\wyscout\data\matches\matches_Italy.json'
with open(path) as f:
    data = json.load(f)

df_matches = pd.DataFrame(data)
df_matches.info()

path = '.\data\wyscout\data\players.json'
with open(path) as f:
    data = json.load(f)

df_players = pd.DataFrame(data)
df_players.info()

path = '.\data\wyscout\data\events\events_Italy.json'
with open(path) as f:
    data = json.load(f)
    
df_events = pd.DataFrame(data)
df_events.info()