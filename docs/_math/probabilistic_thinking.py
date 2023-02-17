# %%
import os
import time
import math
import numpy as np
import scipy as sp
import scipy.stats as spt
import matplotlib.pyplot as plt
import matplotlib as mplt
import pandas as pd
import seaborn as sns


def birthday(m:int) -> int:
    """
    Calculate the number of pairs having duplicate birthdays
    """
    return int(m * (m-1) / (2 * 365))


def plot_birthday():
    num = list(range(1, 1000, 10))
    pair = [birthday(x) for x in num]
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(num, pair, 'k')
    ax.set_title("Number of pairs having same birthday")
    ax.set_xlabel("Population Size")
    

def check_captcha_expectation(m:int, n:int) -> float:
    """
    Calculate the expected number of duplicates based on
        m: sample size
        n: claimed popultion size
    ------
    Return: expected number of duplicates 
    """
    return (m * (m-1)) / (2 * n)


def plot_check_captcha():
    pop_size = 1e6
    sample_size = list(range(1, 2000, 10))
    expected_duplicates = [
        check_captcha_expectation(x, pop_size) for x in sample_size
        ]
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(sample_size, expected_duplicates, 'k')
    ax.set_title("Expected number of duplicates for n = 1,000, 000")
    ax.set_xlabel("Sample Size");
    
    
def markov_prob(a:int, expectation) -> float:
    """
    Calculate the markov inequality (the estimated probability) 
    for checking captchas
        a: bound value
        expectation: the expectation of X - E[X]
    -------
    Return: the estimated probability
    """
    return expectation / a


def plot_markov_cpatcha():
    """
    Plot the markov probability by fixing the sample size
    ---------------
    population size n = 1,000,000
    sample size m = 1000
    duplicate pairs are calculated from check_captcha_expectation
    """
    count_pairs = list(range(1, 100, 5))
    expectation = check_captcha_expectation(1000, 1e6)
    prob = [markov_prob(x, expectation) for x in count_pairs]
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(count_pairs, prob, 'k')
    ax.set_title("Markov probability n=1,000,000, m=1,000")
    ax.set_xlabel("Duplicate Pairs")
    ax.set_ylabel("Markov Inequality Probability")
    ax.set_xticks(np.arange(0, 102, 10))
    print(markov_prob(10, expectation))


def count_by_rounding(n:int) -> list:
    """
    Count by round the formula log_2(1+2^X_n)
    """
    count = []
    vals = []
    for i in range(n):
        x_n = np.log2(1 + i)
        count.append(np.round(x_n))
    
    for c in count:
        theta_n = 2**c -1
        vals.append(theta_n)

    return vals

def morris_count(n:int) -> list:
    """
    Implement morris's approximate counting
    """
    count = []
    count.append(0)
    vals = []
    for i in range(n):
        p = 1/(2**count[i])
        r = np.random.random() 
        if p > r:
            count.append(count[i] + 1)
        else:
            count.append(count[i])

    for c in count[1:]:
        theta_n = 2**c -1
        vals.append(theta_n)

    return vals


def plot_count():
    """
    plot count path for deterministic and stochastic algorithms
    """
    nx = list(range(1000))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(nx, count_by_rounding(1000), 'k:', label='Deterministic')
    axes[0].plot(nx, morris_count(1000), 'k--', label='Stochastic');
    axes[0].plot(nx, nx, 'k-', label='n')
    axes[0].legend()
    axes[1].plot(nx, count_by_rounding(1000), 'k:')
    axes[1].plot(nx, morris_count(1000), 'k--')
    axes[1].plot(nx, nx, 'k')
    axes[1].set_title("Morris' counting path is random")
    
    
def plot_hypergeometric1(case='case-1'):
    # fix n = 1e5
    # fix A = 100
    # plot k = 1, 10, 50
    simulation_cases = {
        'case-1': {'N': 1000, 'A': 10},
        'case-2': {'N': 1000, 'A': 100},
        'case-3': {'N': 1e6, 'A': 1e3}
    }
    
    case = simulation_cases[case]
    N = int(case['N'])
    A = int(case['A'])
    M = list(range(2, 99))
    K = []
    P = []
    
    # construct vector for K
    for x in M:
        temp = math.floor(0.5*x)
        if temp < case['A']:
            K.append(temp)
        else:
            K.append(case['A'])
    
    # calculate the probability
    for i in range(len(M)):
        m_value = M[i]
        k_value = K[i]
        num = math.comb(A, k_value) * math.comb((N-A), (m_value-k_value))
        deno = math.comb(N, m_value)
        p = (num/deno)
        P.append(p)
    
    print(P[-1])
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    axes[0, 0].plot(M, K, 'k')
    axes[0, 0].set_xlabel('m', fontsize=12)
    axes[0, 0].set_ylabel('k', fontsize=12)
    axes[0, 1].plot(M, P, 'k--')
    axes[0, 1].set_ylabel('P', fontsize=12)
    axes[1, 0].set_axis_off()
    axes[1, 0].text(
        0, 0.5, 
        f"N={N}, A={A}, share={int(A/N*100)}%", fontsize=12)
    axes[1, 1].plot(M, P, 'k--')
    axes[1, 1].set_xlim(2, 10)
    axes[1, 1].set_xlabel('m', fontsize=12)
    axes[1, 1].set_ylabel('P', fontsize=12)
    

    
def plot_hypergeometric1(case='case-1'):
    # fix n = 1e5
    # fix A = 100
    # plot k = 1, 10, 50
    simulation_cases = {
        'case-1': {'N': 1000, 'A': 10},
        'case-2': {'N': 1000, 'A': 100},
        'case-3': {'N': 1e6, 'A': 1e3}
    }
    
    case = simulation_cases[case]
    N = int(case['N'])
    A = int(case['A'])
    M = list(range(2, 99))
    K = []
    P = []
    
    # construct vector for K
    for x in M:
        temp = math.floor(0.5*x)
        if temp < case['A']:
            K.append(temp)
        else:
            K.append(case['A'])
    
    # calculate the probability
    for i in range(len(M)):
        m_value = M[i]
        k_value = K[i]
        num = math.comb(A, k_value) * math.comb((N-A), (m_value-k_value))
        deno = math.comb(N, m_value)
        p = (num/deno)
        P.append(p)
    
    print(P[-1])
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    axes[0, 0].plot(M, K, 'k')
    axes[0, 0].set_xlabel('m', fontsize=12)
    axes[0, 0].set_ylabel('k', fontsize=12)
    axes[0, 1].plot(M, P, 'k--')
    axes[0, 1].set_ylabel('P', fontsize=12)
    axes[1, 0].set_axis_off()
    axes[1, 0].text(
        0, 0.5, 
        f"N={N}, A={A}, share={int(A/N*100)}%", fontsize=12)
    axes[1, 1].plot(M, P, 'k--')
    axes[1, 1].set_xlim(2, 10)
    axes[1, 1].set_xlabel('m', fontsize=12)
    axes[1, 1].set_ylabel('P', fontsize=12)
    
    
def plot_hypergeometric2():
    # fix n = 1e5
    # fix A = 100
    # plot k = 1, 10, 50
    simulation_cases = {
        'case-1': {'N': 1000, 'A': 10, 'result': [], 'style':'k-'},
        'case-2': {'N': 1000, 'A': 100, 'result': [], 'style': 'k--'},
        'case-3': {'N': 1000, 'A': 200, 'result': [], 'style': 'k:'}
    }
    
    M = list(range(2, 125))

    for key, item in simulation_cases.items():
        N = int(item['N'])
        A = int(item['A'])
        # calculate the probability
        for m_value in M:
            k_value = 1
            num = math.comb(A, k_value) * math.comb((N-A), (m_value-k_value))
            deno = math.comb(N, m_value)
            p = (num/deno)
            item['result'].append(p)
    
    fig, axes = plt.subplots(1, 1, figsize=(6, 3))
    for key, item in simulation_cases.items():
        axes.plot(M, item['result'], item['style'], label=key)
        print(M[3], item['result'][3])
        print(M[23], item['result'][23])
    axes.set_xlabel('m', fontsize=12)
    axes.set_ylabel('P', fontsize=12)
    axes.legend(
        [
            'N=1000, A= 10, share=1%',
            'N=1000, A=100, share=10%',
            'N=1000, A=200, share=20%'
            ]
        )
    
    
def hypergeom(N, A, m, k):
    num = math.comb(A, k) * math.comb((N-A), (m-k))
    deno = math.comb(N, m)
    p = (num/deno)
    
    return p
    

def chebyshev_inequality():
        
    epsilon1 = 0.1
    epsilon2 = 0.01


    n = list(range(1, 1000))
    p1 = []
    p2 = []

    for x in n:
        p1.append(1/(x * epsilon1 ** 2))
        p2.append(1/(x * epsilon2 ** 2))

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot(n[10:], p1[10:], 'k--', label='epsilon = 0.1')
    ax.plot(n[10:], p2[10:], 'k:', label='epsilon=0.01')
    ax.set_xlabel('Sample size n')
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"})
    ax.set_ylabel(r"Calculation of $\frac{1}{n \epsilon^2)}$")
    ax.set_ylim(0, 100)
    ax.text(100, p1[101]+1, '0.1001')
    ax.text(953, p2[-1]+3, '10.01')
    ax.legend()
    print(p1[-1], p2[-1])


if __name__ == "__main__":
    chebyshev_inequality()
    
    plt.savefig('../images/estimation_sample_size.png', dpi=300, bbox_inches="tight")

    
    
# %%
