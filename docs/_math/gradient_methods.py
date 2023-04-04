# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gradient_descent(x0, alphas, grad, proj=lambda x : x):
    """Project gradient descent.
    This function is written for the purpose of the tutorial.
    
    Input:
        x0: initial point
        alphas: list of step sizes
        grad: gradient function
        proj: projection function
    Output:
        a list of sequence of iterated points
    """
    
    xt = [x0]
    for step in alphas:
        xt.append(proj(xt[-1] - step * grad(xt[-1])))
    return xt


def least_square_sums(A, b, x):
    """
    leaste square objective function that should be minimized
    Input:
        A: matrix, m by n   
        b: vector, m by 1
        x: vector, n by 1
    """
    m, n = A.shape
    return (0.5/m) * np.linalg.norm(A @ x - b) ** 2


def leaste_square_gradient(A, b, x):
    """
    leaste square gradient function (first order derivative)
    Input:
        A: matrix, m by n   
        b: vector, m by 1
        x: vector, n by 1
    """
    m, n = A.shape
    return (1/m) * A.T @ (A @ x - b)


def simulate_data(m, n):
    """
    simulate data for least square problem
    Input:
        m: number of samples
        n: number of features
    Output:
        A: matrix, m by n   
        b: vector, m by 1
        theta: coefficient vector, n by 1
    """
    A = np.random.randn(m, n)
    x = np.random.randn(n, 1)
    noise = np.random.normal(0, 0.1, (m, 1))
    b = A @ x + noise
    return A, b, x 
    

    
    
if __name__ == "__main__":
    # assume m > n
    m = 100  # number of samples
    n = 3  # number of features
    # x is coefficient vector
    A, b, x = simulate_data(m, n)
    # calculate objective function and gradient
    # least square sums that should be minimized
    objective = lambda x: least_square_sums(A, b, x)
    # gradient descent for least square
    gradient = lambda x: leaste_square_gradient(A, b, x)
    
    # initialize x0
    x0 = np.random.normal(0, 1, (n, 1))
    
    # 100 iterations
    xt = gradient_descent(x0, [0.1]*100, gradient)
    
    # plot the error term (the sums of squares error)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot([objective(x) for x in xt], "k" ,label="error term")
    ax.set_title("Gradient Descent for Least Square")
    ax.plot([least_square_sums(A, b, x)]*len(xt), 'k--',
                    label="true error term")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sum of Squares Error") 
    ax.legend()
    # plt.savefig('../math/images/gradient-ols.png',
    #             dpi=300, bbox_inches="tight")
    df = pd.DataFrame.from_dict(
        {
            "Initial guess (x0)": x0.flatten(),
            "True coefficient (x)": x.flatten(),
            "Estimated coefficients (xt)": xt[-1].flatten()
        }
    )
    


# %%
