#%%
import numpy as np
import matplotlib.pyplot as plt


# f(x) = x^{1/2}
def f(x, power=0.5):
    return x**power

# derivative of f(x)
def df(x, n):
    # not a general solution but works for this case
    # assume n >= 1
    # using recursion
    if n == 1:
        return 1/2 * x**(-1/2)
    else:
        return (1/2 - (n-1)) * x**(1/2-n) * df(x, n-1)


def taylor_series(x, n, a=1):
    # n is integer assume n >= 1
    ts = 0
    for i in range(1, n):
        ts += df(a, i) * (x-a)**i / np.math.factorial(i)
    return ts + f(a)


# plot taylor series
def plot_taylor_series():
    n = 17  # set up to 16th order
    nlist = list(range(1, 10))
    y = [taylor_series(2, i) for i in nlist]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(nlist, y, 'k--', label='Taylor series')
    # add np.sqrt(2) as a horizontal line
    ax.axhline(y=np.sqrt(2), color='k', linestyle='-', label='sqrt(2)')
    ax.set_xlabel('order')
    ax.set_ylabel('value')
    ax.set_title('Taylor series of sqrt(2) at x=2 around a=1')
    # legend right bottom
    ax.legend(loc='lower right')
        


if __name__ == "__main__":
    # plot option retinal display
    plt.rcParams['figure.dpi'] = 300
    print("hello world")
    print(f(1))
    print(df(1, 4))
    print(taylor_series(2, 6))
    # retinal display
    %config InlineBackend.figure_format = 'retina'
    plot_taylor_series()
# %%
