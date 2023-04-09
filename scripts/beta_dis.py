# %%
import os
import sys
import itertools
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

# set seaborn style
sns.set_style()


# jax imports
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


# setup memory allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.5'


def plot_beta_dist():
    
    hyperparams = [1, 3, 10]
    alpha_beta = list(itertools.product(hyperparams, hyperparams))
    
    # random variable
    x = np.linspace(0, 1, 1000)
    
    fig, axes = plt.subplots(3, 3, figsize=(9, 8))
    axes = axes.flatten()
    for idx, (alpha, beta) in enumerate(alpha_beta):
        y = sp.stats.beta.pdf(x, alpha, beta)
        axes[idx].plot(x, y, "k-")
        axes[idx].set_ylim(0, 6)
        axes[idx].set_title(f"$\\alpha={alpha}, \\beta$={beta}")
    fig.subplots_adjust(hspace=0.5)
    # plt.savefig('./docs/math/images/beta_dist.png', dpi=300, bbox_inches='tight')
    

def free_throws():
    """Plot binomial distribution for free throws"""
    p_vals = [0.5, 0.7, 0.9]
    markers = ["ko-", "ko--", "ko:"]
    n = 10
    x = np.arange(0, n+1, 1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for idx in range(len(p_vals)):
        p = p_vals[idx]
        y = sp.stats.binom.pmf(x, n, p)
        mk = markers[idx]
        ax.plot(x, y, mk, label=f"$p={p}$")
    # add a vertical line at x=6
    ax.axvline(x=6, color="b", linestyle="-.")
    ax.set_title("Binomial distribution for free throws")
    plt.legend()
    plt.savefig('./docs/math/images/free_throws.png', dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    print(os.getcwd())
    # plt.style.use('default')
    # plt.style.use('seaborn')
    # plot_beta_dist()
    free_throws()

   
# %%
