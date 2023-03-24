import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

import statsmodels.api as sm
import statsmodels.formula.api as smf

import os
import pathlib
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

# load data
path = '..\..\data\wyscout\data\events\events_Italy.json'
with open(path) as f:
    data = json.load(f)

train = pd.DataFrame(data)

shots = train.loc[train['subEventName'] == 'Shot']
# get shot coordinates as separate columns
shots["X"] = shots.positions.apply(lambda cell: (100 - cell[0]['x']) * 105/100)
shots["Y"] = shots.positions.apply(lambda cell: cell[0]['y'] * 68/100)
shots["C"] = shots.positions.apply(
    lambda cell: abs(cell[0]['y'] - 50) * 68/100)
# calculate distance and angle
shots["Distance"] = np.sqrt(shots["X"]**2 + shots["C"]**2)
shots["Angle"] = np.where(np.arctan(7.32 * shots["X"] / (shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)) > 0, np.arctan(7.32 * shots["X"] / (
    shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)), np.arctan(7.32 * shots["X"] / (shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)) + np.pi)
# if you ever encounter problems (like you have seen that model treats 0 as 1 and 1 as 0) while modelling - change the dependant variable to object
shots["Goal"] = shots.tags.apply(
    lambda x: 1 if {'id': 101} in x else 0).astype(object)

# plot pitch
pitch = VerticalPitch(line_color='black', half=True, pitch_type='custom',
                      pitch_length=105, pitch_width=68, line_zorder=2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
# subtracting x from 105 but not y from 68 because of inverted Wyscout axis
# calculate number of shots in each bin
bin_statistic_shots = pitch.bin_statistic(105 - shots.X, shots.Y, bins=50)
# make heatmap
pcm = pitch.heatmap(bin_statistic_shots,
                    ax=ax["pitch"], cmap='Reds', edgecolor='white', linewidth=0.01)
# make legend
ax_cbar = fig.add_axes((0.95, 0.05, 0.04, 0.8))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Shot map - 2017/2018 Seria A Season', fontsize=30)
plt.show()

# take only goals
goals = shots.loc[shots["Goal"] == 1]
# plot pitch
pitch = VerticalPitch(line_color='black', half=True, pitch_type='custom',
                      pitch_length=105, pitch_width=68, line_zorder=2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
# calculate number of goals in each bin
bin_statistic_goals = pitch.bin_statistic(105 - goals.X, goals.Y, bins=50)
# plot heatmap
pcm = pitch.heatmap(bin_statistic_goals,
                    ax=ax["pitch"], cmap='Reds', edgecolor='white')
# make legend
ax_cbar = fig.add_axes((0.95, 0.05, 0.04, 0.8))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Goal map - 2017/2018 Seria A Season', fontsize=30)
plt.show()

# plot pitch
pitch = VerticalPitch(line_color='black', half=True, pitch_type='custom',
                      pitch_length=105, pitch_width=68, line_zorder=2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
bin_statistic = pitch.bin_statistic(105 - shots.X, shots.Y, bins=50)
# normalize number of goals by number of shots
bin_statistic["statistic"] = bin_statistic_goals["statistic"] / \
    bin_statistic["statistic"]
# plot heatmap
pcm = pitch.heatmap(
    bin_statistic, ax=ax["pitch"], cmap='Reds', edgecolor='white', vmin=0, vmax=0.6)
# make legend
ax_cbar = fig.add_axes((0.95, 0.05, 0.04, 0.8))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Probability of scoring', fontsize=30)
plt.show()

# creating extra variables
shots["X2"] = shots['X']**2
shots["C2"] = shots['C']**2
shots["AX"] = shots['Angle']*shots['X']

# list the model variables you want here
model_variables = ['Angle', 'Distance', 'X', 'C', "X2", "C2", "AX"]
model = ''
for v in model_variables[:-1]:
    model = model + v + ' + '
model = model + model_variables[-1]

# fit the model
test_model = smf.glm(formula="Goal ~ " + model, data=shots,
                     family=sm.families.Binomial()).fit()
# print summary
print(test_model.summary())
b = test_model.params

# return xG value for more general model


def calculate_xG(sh):
    bsum = b[0]
    for i, v in enumerate(model_variables):
        bsum = bsum+b[i+1]*sh[v]
    xG = 1/(1+np.exp(bsum))
    return xG


# add an xG to my dataframe
xG = shots.apply(calculate_xG, axis=1)
shots = shots.assign(xG=xG)

# Create a 2D map of xG
pgoal_2d = np.zeros((68, 68))
for x in range(68):
    for y in range(68):
        sh = dict()
        a = np.arctan(7.32 * x / (x**2 + abs(y-68/2)**2 - (7.32/2)**2))
        if a < 0:
            a = np.pi + a
        sh['Angle'] = a
        sh['Distance'] = np.sqrt(x**2 + abs(y-68/2)**2)
        sh['D2'] = x**2 + abs(y-68/2)**2
        sh['X'] = x
        sh['AX'] = x*a
        sh['X2'] = x**2
        sh['C'] = abs(y-68/2)
        sh['C2'] = (y-68/2)**2

        pgoal_2d[x, y] = calculate_xG(sh)

# plot pitch
pitch = VerticalPitch(line_color='black', half=True, pitch_type='custom',
                      pitch_length=105, pitch_width=68, line_zorder=2)
fig, ax = pitch.draw()
# plot probability
pos = ax.imshow(pgoal_2d, extent=[-1, 68, 68, -1], aspect='auto',
                cmap=plt.cm.Reds, vmin=0, vmax=0.3, zorder=1)
fig.colorbar(pos, ax=ax)
# make legend
ax.set_title('Probability of goal')
plt.xlim((0, 68))
plt.ylim((0, 60))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
