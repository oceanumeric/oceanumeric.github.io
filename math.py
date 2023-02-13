# %%
import os
import time
import numpy as np
import scipy as sp
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



if __name__ == "__main__":
    plot_markov_cpatcha()
    
    
    
    
    
    
    
    
    
# %%
