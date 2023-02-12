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
    plt.grid()


if __name__ == "__main__":
    print('hello')