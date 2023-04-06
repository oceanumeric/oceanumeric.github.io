# %% 
import os
import mmh3 
import numpy as np
import matplotlib.pyplot as plt


def simulate_bitmap(sim_n = 100):
    """
    Simulate the bitmap pattern from the paper by Flajolet and Martion (1985)
    run sim_n (default = 100) times of loops
        hash the index integer
        calculate the number of trailing zeros 
    
    return the probability of have 3 consecutive trailing zeros in sim_n times
    """

    distinct_count = 0 

    k_3zeros = 0

    z = 0
    for i in range(sim_n):
        # use murmurhash function 
        hash_int = mmh3.hash(str(i), signed = False)
        hash_str = "{0:b}".format(hash_int)
        count_trailing0s = _calculate_trailing0s(hash_str) 
        if count_trailing0s == 3:
            k_3zeros += 1
    return k_3zeros / sim_n


def _calculate_trailing0s(bit_str, k = 0):
    """
    A recursive function
    WARNing_ you need return for both branches !!!
    """

    if bit_str[-1] == '1':
        return k
    else:
        k += 1
        return _calculate_trailing0s(bit_str[:-1], k)
    

def plot_bitmap_pattern():

    sims = np.linspace(100, 1e5, 30, dtype=int)
    sims_prob = []
    for sn in sims:
        prob = simulate_bitmap(sn)
        sims_prob.append(prob)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(sims, sims_prob, 'k--')
    ax[0].scatter(sims[-1], sims_prob[-1], edgecolor='k', facecolor='none')
    ax[0].text(sims[-1]-10000, sims_prob[-1]+0.001, "0.06153")
    ax[0].set_title("Linear Scale")
    ax[0].set_xlabel('Number of distinct values')
    ax[0].set_ylabel("Prob")
    ax[1].plot(sims, sims_prob, 'k--')
    ax[1].scatter(sims[-1], sims_prob[-1], edgecolor='k', facecolor='none')
    ax[1].text(sims[-1]-10000, sims_prob[-1]+0.001, "0.06153")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title("Log-Log Scale")
    ax[1].set_xlabel('Number of distinct values')



def fm_algorithm(num = 1000):
    """
    plot the probabilistic distinct elements counting algorithms
    """
    bitmap = '0'*32
    z = 0
    for i in range(num):
        hash_int = mmh3.hash(str(i), signed=False)
        hash_str = "{0:b}".format(hash_int)
        count_trailing0s = _calculate_trailing0s(hash_str)

        bitmap = bitmap[:count_trailing0s] + '1' + bitmap[count_trailing0s+1:]

    r = bitmap.find('0')
    
    return 2**r 


def plot_fm_algorithm():

    nrange = np.linspace(100, 1e5, 30, dtype=int)
    estimation = []

    for nn in nrange:
        estim = fm_algorithm(nn)
        estimation.append(estim)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(nrange, estimation, 'k--')
    ax.plot(nrange, nrange, 'k')




if __name__ == "__main__":
    print(os.getcwd())

    # print(fm_algorithm(num=83678))
    
    # plot_bitmap_pattern()

    # plt.savefig('./docs/math/images/fm_sim_plot.png', dpi=300, bbox_inches="tight")

    plot_fm_algorithm()
    plt.savefig('./docs/math/images/fm_count_plot.png', dpi=300, bbox_inches="tight")





# %%
