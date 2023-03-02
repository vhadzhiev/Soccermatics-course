import matplotlib.pyplot as plt
from mplsoccer import Pitch, Sbopen

parser = Sbopen()
df, related, freeze, tactics = parser.event(7576)

# get the passes whithout throw-ins
passes = df.loc[df['type_name'] == 'Pass'].loc[df['sub_type_name'] != 'Throw-in'].set_index('id')

# Option 1: plot the passes by looping through them.
pitch = Pitch(pitch_type='statsbomb', line_color="black", pitch_length=120, pitch_width=80)
fig, ax = pitch.draw(figsize=(20, 14))

for i, thepass in passes.iterrows():
    # if pass made by Cristiano Ronaldo
    if thepass['player_name'] == 'Cristiano Ronaldo dos Santos Aveiro':
        x = thepass['x']
        y = thepass['y']
        # plot circle
        passCircle = plt.Circle((x, y), 2, color="red")
        passCircle.set_alpha(.2)
        ax.add_patch(passCircle)
        dx = thepass['end_x'] - x
        dy = thepass['end_y'] - y
        # plot arrow
        passArrow = plt.Arrow(x, y, dx, dy, width=3, color="red")
        ax.add_patch(passArrow)

ax.set_title("Cristiano Ronaldo passes against Spain", fontsize=24)
fig.set_size_inches(20, 14)
plt.show()

# Option 2: plot the passes using mplsoccer functions
mask_ronaldo = (df.type_name == 'Pass') & (df.sub_type_name != 'Throw-in') & (
        df.player_name == 'Cristiano Ronaldo dos Santos Aveiro')
df_pass = df.loc[mask_ronaldo, ['x', 'y', 'end_x', 'end_y']]

pitch = Pitch(pitch_type='statsbomb', line_color="black", pitch_length=120, pitch_width=80)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pitch.arrows(df_pass.x, df_pass.y,
             df_pass.end_x, df_pass.end_y, color="red", ax=ax['pitch'])
pitch.scatter(df_pass.x, df_pass.y, alpha=0.2, s=500, color="red", ax=ax['pitch'])
fig.suptitle("Cristiano Ronaldo passes against Spain", fontsize=30)
plt.show()

# Plotting multiple pass maps on one figure
# prepare the dataframe of passes by Portugal that were no-throw ins
mask_portugal = (df.type_name == 'Pass') & (df.team_name == "Portugal") & (df.sub_type_name != "Throw-in")
df_passes = df.loc[mask_portugal, ['x', 'y', 'end_x', 'end_y', 'player_name']]
# get the list of all players who made a pass
names = df_passes['player_name'].unique()

# draw 4x4 pitches
pitch = Pitch(pitch_type='statsbomb', line_color="black", pitch_length=120, pitch_width=80, pad_top=20)
fig, axs = pitch.grid(ncols=4, nrows=4, grid_height=0.85, title_height=0.06, axis=False,
                      endnote_height=0.04, title_space=0.04, endnote_space=0.01)

# for each player
for name, ax in zip(names, axs['pitch'].flat[:len(names)]):
    # put player name over the plot
    ax.text(60, -10, name,
            ha='center', va='center', fontsize=14)
    # take only passes by this player
    player_df = df_passes.loc[df_passes["player_name"] == name]
    # scatter
    pitch.scatter(player_df.x, player_df.y, alpha=0.2, s=50, color="red", ax=ax)
    # plot arrow
    pitch.arrows(player_df.x, player_df.y,
                 player_df.end_x, player_df.end_y, color="red", ax=ax, width=1)

# We have more than enough pitches - remove them
for ax in axs['pitch'][-1, 16 - len(names):]:
    ax.remove()

# Another way to set title using mplsoccer
axs['title'].text(0.5, 0.5, 'Portugal passes against Spain', ha='center', va='center', fontsize=30)
fig.set_size_inches(20, 14)
plt.show()
