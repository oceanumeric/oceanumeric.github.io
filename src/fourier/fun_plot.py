import numpy as np
import matplotlib.pyplot as plt 


def plot_sin_1_x():
    x = np.arange(0, 0.3, 0.0001)
    y = np.sin(1/x)

    x1 = np.arange(0, 6, 0.001)
    y1 = np.sin(1/x1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), facecolor='#fffff8')
    axs[0].plot(x, y)
    axs[0].axhline(y=0, color='k')
    axs[0].grid()
    axs[0].set_title('Axis (0, 0.3]')
    axs[1].plot(x1, y1)
    axs[1].axhline(y=0, color='k')
    axs[1].grid()
    axs[1].set_title('Axis (0, 6]')
    fig.suptitle(r"Plot of function $\sin(1/x)$")
    fig.savefig('docs/math/fourier/images/sinfun.png', dpi=600, bbox_inches='tight')
    
    
if __name__ == "__main__":
    plot_sin_1_x()