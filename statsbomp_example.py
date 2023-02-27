# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 10:22:14 2023

@author: USER
"""

# Sbopen loads data from the StatsBomb open-data
from mplsoccer import Sbopen
parser = Sbopen()

# check the available competitions
df_competition = parser.competition()
df_competition.info()

# select games from WC 2018
df_match = parser.match(competition_id=43, season_id=3)
df_match.info()

# select lineups from Portugal - Spain match
df_lineup= parser.lineup(7576)
df_lineup.info()

# select additional info for the match
df_event, df_related, df_freeze, df_tactics = parser.event(7576)
df_event.info()
df_related.info()
df_freeze.info()
df_tactics.info()

# 360 data not available yet, so just picked a random match
df_frame, df_visible = parser.frame(3788741)
df_frame.info()
df_visible.info()
