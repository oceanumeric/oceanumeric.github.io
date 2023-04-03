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


def qr_iteration(A:np.ndarray, num_simulations:int):
    """Returns all eigenpairs of A.

    Parameters
    ----------
    A : ndarray
        Square matrix.
    num_simulations : int
        Number of QR iterations to perform.

    Returns
    -------
    v : ndarray
        all eigenvectors.
    lambda : float
        all eigenvalues.
    """
 
    for _ in range(num_simulations):
        Q, R = np.linalg.qr(A)
        A = R @ Q

    v = Q
    lambda_all = A.diagonal()
    return v, lambda_all


if __name__ == '__main__':
    np.random.seed(789)
    print(np.random.rand(4, 4))
    A = np.random.rand(4, 4)
    v, lambda_top = power_iteration(A, 40)
    print('v =', v)
    print('lambda =', lambda_top)
    print('result from numpy:', np.linalg.eig(A)[0][0])
    v_all, lambda_all = qr_iteration(A, 40)
    print('v_all =', v_all, '\n')
    print('lambda_all =', lambda_all, '\n')
    v_temp
    print("result from numpy:", )
