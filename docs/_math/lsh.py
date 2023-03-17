# %% 
import os
import mmh3 
import numpy as np
import matplotlib.pyplot as plt


def plot_lsh():
    """
    Plot LSH based on [1- (1-p^r)^b]
    """
    
    r = 3 
    p1 = 0.4
    p2 = 0.7
    b_values = list(range(1, 51))
    
    def _cal(p, r, b):
        return (1-(1-p**r)**b)
    
    pr_values1 = [_cal(p1, r, x) for x in b_values]
    pr_values2 = [_cal(p2, r, x) for x in b_values]
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot(b_values, pr_values1, 'k:', label = "p = 0.4")
    ax.plot(b_values, pr_values2, 'k--', label = "p = 0.7")
    ax.legend()
    ax.set_xlabel("size of bands")
    ax.set_ylabel("Pr(matched pair in one block)")
    
    
    
    
    


if __name__ == "__main__":
    print(os.getcwd())
    os.chdir("/home/zou/Github/mysite")
    
    plot_lsh()
    plt.savefig('./docs/math/images/lsh_plot1.png', dpi=300, bbox_inches="tight")
# %%
