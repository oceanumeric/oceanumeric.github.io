import numpy as np
import scipy as sp

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


def qr_iteration_with_hessenberg(A:np.ndarray, num_simulations:int):


    H = sp.linalg.hessenberg(A)
    for _ in range(num_simulations):
        Q, R = np.linalg.qr(H)
        H = R @ Q

    v = Q
    lambda_all = H.diagonal()
    return v, lambda_all


def wilkinson_shift(a, b, c):
    # Calculate Wilkinson's shift for symmetric matrices: 
    delta = (a-c)/2
    shift = c - np.sign(delta)*b**2/(np.abs(delta) + np.sqrt(delta**2+b**2))
    return shift


def qr_with_shift(A:np.ndarray, num_iterations:int):
    
    n, m = A.shape
    eigen_values = []
    if n != m:
        raise ValueError('A must be a symmetric matrix.')
    
    I = np.eye(n)
    
    H = sp.linalg.hessenberg(A)
    for _ in range(num_iterations):
        u = wilkinson_shift(H[n-2, n-2], H[n-1, n-1], H[n-2, n-1])
        Q, R = np.linalg.qr(H - u*I)
        H = R @ Q + u*I
    
    v = Q
    lambda_all = H.diagonal()
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
    print("result from numpy:", np.linalg.eig(A))
    v_all1, lambda_all1 = qr_iteration_with_hessenberg(A, 40)
    print('v_all1 =', v_all1, '\n')
    print('lambda_all1 =', lambda_all1, '\n')
    print(qr_with_shift(A, 40))
