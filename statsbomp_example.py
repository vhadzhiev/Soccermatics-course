# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 10:22:14 2023

@author: USER
"""

# Sbopen loads data from the StatsBomb open-data
from mplsoccer import Sbopen
parser = Sbopen()

df_competition = parser.competition()
df_competition.info()

df_match = parser.match(competition_id=72, season_id=30)
df_match.info()

df_lineup= parser.lineup(69301)
df_lineup.info()

df_event, df_related, df_freeze, df_tactics = parser.event(69301)
df_event.info()
df_related.info()
df_freeze.info()
df_tactics.info()

df_frame, df_visible = parser.frame(3788741)
df_frame.info()
df_visible.info()
