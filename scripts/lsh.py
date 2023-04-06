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
    

def plot_lsh2():
    
    p = np.linspace(0, 1, 100)
    
    b1, r1 = 5, 3
    b2, r2 = 5, 7
    b3, r3 = 7, 3
    b4, r4 = 7, 7
    
    def _cal(p, r, b):
        return (1-(1-p**r)**b)
    
    res1 = [_cal(x, b1, r1) for x in p]
    res2 = [_cal(x, b2, r2) for x in p]
    res3 = [_cal(x, b3, r3) for x in p]
    res4 = [_cal(x, b4, r4) for x in p]
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot(p, res1, 'k--', label = f"({b1}, {r1})")
    ax.plot(p, res2, 'k-.', label = f"({b2}, {r2})")
    ax.plot(p, res3, 'k:', label = f"({b3}, {r3})")
    ax.plot(p, res4, 'k-', label = f"({b4}, {r4})")
    ax.plot([0.8]*40, np.linspace(0, 1, 40), '+', color = "#808080")
    ax.plot(p, 0*p+0.8, '+', color = "#808080")
    ax.legend()
    
    
    
    
    


if __name__ == "__main__":
    print(os.getcwd())
    os.chdir("/home/zou/Github/mysite")
    
    plot_lsh2()
    plt.savefig('./docs/math/images/lsh_plot2.png', dpi=300, bbox_inches="tight")
# %%
