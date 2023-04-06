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
    

def case_study1():
    # assume m > n
    m = 100  # number of samples
    n = 3 # number of features
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
    ax.set_yscale("log")
    ax.set_title("Gradient Descent for Least Square")
    ax.plot([least_square_sums(A, b, x)]*len(xt), 'k--',
                    label="true error term")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sum of Squares Error (log scale)") 
    ax.legend()
    plt.savefig('../math/images/gradient-ols.png',
                dpi=300, bbox_inches="tight")
    df = pd.DataFrame.from_dict(
        {
            "Initial guess (x0)": x0.flatten(),
            "True coefficient (x)": x.flatten(),
            "Estimated coefficients (xt)": xt[-1].flatten()
        }
    )
    
    print(df.to_markdown())


def case_study2():
    m, n = 100, 1000
    A = np.random.normal(0, 1, (m, n))
    b = np.random.normal(0, 1, m)
    # The least norm solution is given by the pseudo-inverse
    x_opt = np.linalg.pinv(A.T @ A) @ A.T @ b
    objective = lambda x: least_square_sums(A, b, x)
    gradient = lambda x: leaste_square_gradient(A, b, x)
    
    x0 = np.random.normal(0, 1, n)
    xs = gradient_descent(x0, [0.1]*100, gradient)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot([objective(x) for x in xs], "k", label="error term")
    ax.set_yscale("log")
    ax.plot([least_square_sums(A, b, x_opt)]*len(xs), 'k--',
                        label="true error term")
    ax.set_title("Gradient Descent for Least Square")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sum of Squares Error (log scale)") 
    ax.legend()
    plt.savefig('../math/images/gradient-ols2.png',
                dpi=300, bbox_inches="tight")
    df = pd.DataFrame.from_dict(
        {
            "Initial guess (x0)": x0.flatten(),
            "True coefficient (x)": x_opt.flatten(),
            "Estimated coefficients (xt)": xs[-1].flatten()
        }
    )
    
    print(df.head().to_markdown())
    

def least_square_sums_l2(A, b, x, lam):
    """
    leaste square objective function that should be minimized
    Input:
        A: matrix, m by n   
        b: vector, m by 1
        x: vector, n by 1
        lam: regularization parameter
    """
    m, n = A.shape
    return least_square_sums(A, b, x) + lam/2 * np.linalg.norm(x) ** 2


def least_square_gradient_l2(A, b, x, lam):
    """
    leaste square gradient function (first order derivative)
    Input:
        A: matrix, m by n   
        b: vector, m by 1
        x: vector, n by 1
        lam: regularization parameter
    """
    m, n = A.shape
    return leaste_square_gradient(A, b, x) + lam * x


def case_study3():
    np.random.seed(1337)
    m = 100
    n = 1000
    A = np.random.normal(0, 1, (m, n))
    b = np.random.normal(0, 1, m)
    lam = 0.1
    # the optimal solution is given by the closed form solution
    x_opt = np.linalg.pinv(A.T @ A + lam * np.eye(n)) @ A.T @ b
    objective = lambda x: least_square_sums_l2(A, b, x, lam)
    gradient = lambda x: least_square_gradient_l2(A, b, x, lam)
    
    x0 = np.random.normal(0, 1, n)
    xs = gradient_descent(x0, [0.1]*500, gradient)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot([objective(x) for x in xs], "k", label="error term")
    ax.set_yscale("log")
    ax.plot([least_square_sums_l2(A, b, x_opt, lam)]*len(xs), 'k--',
                        label="true error term")
    ax.set_title("Gradient Descent for Least Square")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sum of Squares Error (log scale)") 
    ax.legend()
    plt.savefig('../math/images/gradient-ols3.png',
                dpi=300, bbox_inches="tight")
    df = pd.DataFrame.from_dict(
        {
            "Initial guess (x0)": x0.flatten(),
            "True coefficient (x)": x_opt.flatten(),
            "Estimated coefficients (xt)": xs[-1].flatten()
        }
    )
    
    print(df.head().to_markdown())
    
    
def frank_wolfe_descent(x0, alphas, grad):
    
    n, _ = x0.shape
    xt = [x0]
    
    # construct the domain of x
    foo = np.linspace(-2, 2, 1000).reshape(-1, 1)
    foo2 = np.hstack([foo]*3)
    x_domain = foo2.T
    for step in alphas:
        foo = grad(xt[-1]).T @ x_domain
        x_tilde = np.amin(np.abs(foo))
        xt.append(xt[-1] + step * (x_tilde - xt[-1]))
    return xt
        
    
def case_study4():
    np.random.seed(1337)
    # assume m > n
    m = 100  # number of samples
    n = 3 # number of features
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
    xt_frank_wolfe = frank_wolfe_descent(x0, [0.1]*100, gradient)
    
    # plot the error term (the sums of squares error)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot([objective(x) for x in xt], "k" ,label="error term")
    ax.plot([objective(x) for x in xt_frank_wolfe], "k:",
                            label="error term (frank wolfe)")
    ax.set_yscale("log")
    ax.set_title("Gradient Descent for Least Square")
    ax.plot([least_square_sums(A, b, x)]*len(xt), 'k--',
                    label="true error term")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sum of Squares Error (log scale)") 
    ax.legend()
    # plt.savefig('../math/images/gradient-ols.png',
    #             dpi=300, bbox_inches="tight")
    df = pd.DataFrame.from_dict(
        {
            "Initial guess (x0)": x0.flatten(),
            "True coefficient (x)": x.flatten(),
            "Estimated coefficients (xt)": xt[-1].flatten(),
            "Estimated coefficients (xt_frank_wolfe)": xt_frank_wolfe[-1].flatten()
        }
    )
    
    print(df.to_markdown())

    
    

    
if __name__ == "__main__":
    print("Hello world!")
    case_study4()
    
    
    


# %%
