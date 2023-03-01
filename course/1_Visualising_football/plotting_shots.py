import matplotlib.pyplot as plt
from mplsoccer import Pitch, Sbopen, VerticalPitch

parser = Sbopen()
# getting info for the Portugal - Spain game
df, related, freeze, tactics = parser.event(7576)
# getting team names
team1, team2 = df.team_name.unique()
# getting dataframe of shots
shots = df.loc[df['type_name'] == 'Shot'].set_index('id')

# Option 1: plot the shots by looping through them.
# as we use Statsbomb, pitch size is in yards
pitch = Pitch(pitch_type='statsbomb', line_color="black", pitch_length=120, pitch_width=80)
fig, ax = pitch.draw(figsize=(20, 14))

for i, shot in shots.iterrows():
    # get the information
    x = shot['x']
    y = shot['y']
    goal = shot['outcome_name'] == 'Goal'
    team_name = shot['team_name']
    # set circle size
    circleSize = 2
    # plot Portugal
    if team_name == team1:
        if goal:
            shotCircle = plt.Circle((x, y), circleSize, color="red")
            plt.text(x + 1, y - 2, shot['player_name'])
        else:
            shotCircle = plt.Circle((x, y), circleSize, color="red")
            shotCircle.set_alpha(.2)
    # plot Spain
    else:
        if goal:
            shotCircle = plt.Circle((pitch.pitch_length - x, pitch.pitch_width - y), circleSize, color="blue")
            plt.text(pitch.pitch_length - x + 1, pitch.pitch_width - y - 2, shot['player_name'])
        else:
            shotCircle = plt.Circle((pitch.pitch_length - x, pitch.pitch_width - y), circleSize, color="blue")
            shotCircle.set_alpha(.2)
    ax.add_patch(shotCircle)
# set title
fig.suptitle("Portugal (red) and Spain (blue) shots", fontsize=24)
fig.set_size_inches(20, 14)
plt.show()

# Option 2: plot the shots using mplsoccer’s Pitch class
pitch = Pitch(pitch_type='statsbomb', line_color='black', pitch_length=120, pitch_width=80)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
# query
mask_portugal = (df.type_name == 'Shot') & (df.team_name == team1)
# finding rows in the df and keeping only necessary columns
df_portugal = df.loc[mask_portugal, ['x', 'y', 'outcome_name', "player_name"]]

# plot them - if shot ended with Goal - alpha 1 and add name
# for Portugal
for i, row in df_portugal.iterrows():
    if row["outcome_name"] == 'Goal':
        # make circle
        pitch.scatter(row.x, row.y, alpha=1, s=500, color="red", ax=ax['pitch'])
        pitch.annotate(row["player_name"], (row.x + 1, row.y - 2), ax=ax['pitch'], fontsize=12)
    else:
        pitch.scatter(row.x, row.y, alpha=0.2, s=500, color="red", ax=ax['pitch'])

mask_spain = (df.type_name == 'Shot') & (df.team_name == team2)
df_spain = df.loc[mask_spain, ['x', 'y', 'outcome_name', "player_name"]]

# for Spain, we need to revert coordinates
for i, row in df_spain.iterrows():
    if row["outcome_name"] == 'Goal':
        pitch.scatter(pitch.pitch_length - row.x, pitch.pitch_width - row.y, alpha=1, s=500, color="blue",
                      ax=ax['pitch'])
        pitch.annotate(row["player_name"], (pitch.pitch_length - row.x + 1, pitch.pitch_width - row.y - 2),
                       ax=ax['pitch'], fontsize=12)
    else:
        pitch.scatter(pitch.pitch_length - row.x, pitch.pitch_width - row.y, alpha=0.2, s=500, color="blue",
                      ax=ax['pitch'])

fig.suptitle("Portugal (red) and Spain (blue) shots", fontsize=30)
plt.show()

# Plotting shots on one half
pitch = VerticalPitch(line_color='black', half=True)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
# plotting all shots
pitch.scatter(df_portugal.x, df_portugal.y, alpha=1, s=500, color="red", ax=ax['pitch'], edgecolors="black")
fig.suptitle("Portugal shots against Spain", fontsize=30)
plt.show()
