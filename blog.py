# %% 
# plot a figure to show the probability of flipping a coin
import numpy as np
import matplotlib.pyplot as plt

# create an array as the time of flipping a coin
x = np.arange(0, 100, 1)

def prob_fun(n):
    # the probability of flipping a coin
    y = 0.5 ** n
    
    return y

fig, ax = plt.subplots()
ax.plot(x, prob_fun(x))

# %%
