import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

num_obs = 20
laliga_df = pd.read_csv("playerstats.txt", delimiter=',')
minutes_model = pd.DataFrame()
minutes_model = minutes_model.assign(minutes=laliga_df['Min'][0:num_obs])
minutes_model = minutes_model.assign(age=laliga_df['Age'][0:num_obs])

# Make an age squared column so we can fit polynomial model.
minutes_model = minutes_model.assign(age_squared=np.power(laliga_df['Age'][0:num_obs], 2))

# plotting the data
fig, ax = plt.subplots(num=1)
ax.plot(minutes_model['age'], minutes_model['minutes'], linestyle='none', marker='.', markersize=10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15, 40))
plt.ylim((0, 3000))
plt.show()

# fitting the model
model_fit = smf.ols(formula='minutes  ~ age   ', data=minutes_model).fit()
print(model_fit.summary())
b = model_fit.params

# comparing the fit
# First plot the data as previously
fig, ax = plt.subplots(num=1)
ax.plot(minutes_model['age'], minutes_model['minutes'], linestyle='none', marker='.', markersize=10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15, 40))
plt.ylim((0, 3000))

# Now create the line through the data
x = np.arange(40, step=1)
y = np.mean(minutes_model['minutes']) * np.ones(40)
ax.plot(x, y, color='black')

# Show distances to line for each point
for i, a in enumerate(minutes_model['age']):
    ax.plot([a, a], [minutes_model['minutes'][i], np.mean(minutes_model['minutes'])], color='red')
plt.show()

# fitting quadratic model
# First fit the model
model_fit = smf.ols(formula='minutes  ~ age + age_squared  ', data=minutes_model).fit()
print(model_fit.summary())
b = model_fit.params

# Compare the fit
fig, ax = plt.subplots(num=1)
ax.plot(minutes_model['age'], minutes_model['minutes'], linestyle='none', marker='.', markersize=10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15, 40))
plt.ylim((0, 3000))
x = np.arange(40, step=1)
y = b[0] + b[1] * x + b[2] * x * x
ax.plot(x, y, color='black')

for i, a in enumerate(minutes_model['age']):
    ax.plot([a, a], [minutes_model['minutes'][i], b[0] + b[1] * a + b[2] * a * a], color='red')
plt.show()
