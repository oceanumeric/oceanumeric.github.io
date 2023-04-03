import numpy as np


def power_iteration(A:np.ndarray, num_simulations:int):
    """Returns the dominant eigenpair of A.

    Parameters
    ----------
    A : ndarray
        Square matrix.
    num_simulations : int
        Number of power iterations to perform.

    Returns
    -------
    v : ndarray
        Dominant eigenvector.
    lambda : float
        Dominant eigenvalue.
    """
    n, m = A.shape
    if n != m:
        raise ValueError('A must be a square matrix.')
    
    # generate a random vector
    v = np.random.rand(n)
    v /= np.linalg.norm(v)

    for _ in range(num_simulations):
        Av = A @ v
        v = Av / np.linalg.norm(Av)

    lambda_top = v.T @ A @ v
    return v, lambda_top



if __name__ == '__main__':
    np.random.seed(789)
    print(np.random.rand(4, 4))
    A = np.random.rand(4, 4)
    v, lambda_top = power_iteration(A, 40)
    print('v =', v)
    print('lambda =', lambda_top)
    print('result from numpy:', np.linalg.eig(A)[0][0])
