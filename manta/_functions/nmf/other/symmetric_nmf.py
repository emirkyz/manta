"""
Symmetric Nonnegative Matrix Factorization (symNMF) using coordinate descent.

Python implementation of the algorithm described in:
A. Vandaele, N. Gillis, Q. Lei, K. Zhong and I.S. Dhillon, "Efficient 
and Non-Convex Coordinate Descent for Symmetric Nonnegative Matrix 
Factorization", IEEE Trans. on Signal Processing 64 (21), pp. 5571-5584, 2016.

This solves the problem:
    min_{H >= 0}  1/2 * ||A - HH^T||_F^2

where A is an n-by-n symmetric nonnegative matrix and H is n-by-r.
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple, Union
from scipy.sparse import issparse, csc_matrix


def cubic_root(d: float) -> float:
    """Compute the cubic root of d, handling negative values."""
    if d < 0.0:
        return -cubic_root(-d)
    else:
        return np.power(d, 1.0/3.0)


def best_polynomial_root(a: float, b: float) -> float:
    """
    Solve the problem: min_{x>=0} x^3 + ax + b
    This is the core optimization step in the coordinate descent algorithm.
    """
    a3 = 4 * a**3
    b2 = 27 * b**2
    delta = a3 + b2
    
    if delta <= 0:  # 3 distinct real roots or 1 real multiple solution
        r3 = 2 * np.sqrt(-a/3)
        th3 = np.arctan2(np.sqrt(-delta/108), -b/2) / 3
        ymax = 0
        xopt = 0
        
        for k in [0, 2, 4]:
            x = r3 * np.cos(th3 + (k * np.pi / 3))
            if x >= 0:
                y = x**4/4 + a*x**2/2 + b*x
                if y < ymax:
                    ymax = y
                    xopt = x
        return xopt
    else:  # 1 real root and two complex
        z = np.sqrt(delta/27)
        x = cubic_root(0.5*(-b + z)) + cubic_root(0.5*(-b - z))
        y = x**4/4 + a*x**2/2 + b*x
        if y < 0 and x >= 0:
            return x
        else:
            return 0.0


def precompute_norms(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute norms and inner products for efficient updates.
    
    Returns:
        nL: norm of each row of H (n,)
        nC: norm of each column of H (r,)  
        HH: H^T @ H (r, r)
    """
    n, r = H.shape
    
    # Row norms: ||H[i,:]||^2
    nL = np.sum(H**2, axis=1)
    
    # Column norms: ||H[:,j]||^2
    nC = np.sum(H**2, axis=0)
    
    # Gram matrix: H^T @ H
    HH = H.T @ H
    
    return nL, nC, HH


def symmetric_nmf(A: np.ndarray, 
                  r: Union[int, np.ndarray], 
                  options: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Symmetric Nonnegative Matrix Factorization using coordinate descent.
    
    Args:
        A: (n, n) symmetric nonnegative matrix to factorize
        r: Either the factorization rank (int) or initial matrix H (n, r)
        options: Dictionary with optional parameters:
            - maxiter: Maximum number of iterations (default: 100)
            - timelimit: Maximum time in seconds (default: 5.0)
            - display: Whether to show progress ('on'/'off', default: 'on')
            - shuffle_columns: Whether to shuffle column order (default: False)
            - init_matrix: Initialization method ('zeros'/'random', default: 'zeros')
            - seed: Random seed for reproducibility (default: None)
    
    Returns:
        H: (n, r) nonnegative matrix such that A ≈ H @ H.T
        errors: Array of objective function values over iterations
        times: Array of computation times over iterations
    """
    # Validate input
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D numpy array")
    
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A must be square")
    
    # Set default options
    if options is None:
        options = {}
    
    maxiter = options.get('maxiter', 100)
    timelimit = options.get('timelimit', 5.0)
    display = options.get('display', 'on')
    shuffle_columns = options.get('shuffle_columns', False)
    init_matrix = options.get('init_matrix', 'zeros')
    seed = options.get('seed', None)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize H
    if isinstance(r, int):
        # r is the factorization rank
        if init_matrix == 'zeros':
            H = np.zeros((n, r))
        elif init_matrix == 'random':
            H = np.random.rand(n, r)
        else:
            raise ValueError("init_matrix must be 'zeros' or 'random'")
    else:
        # r is the initial matrix H
        H = r.copy().astype(float)
        r = H.shape[1]
        if H.shape[0] != n:
            raise ValueError("Initial H must have the same number of rows as A")
    
    # Validate parameters
    if maxiter <= 0 or not isinstance(maxiter, int):
        raise ValueError("maxiter must be a positive integer")
    if timelimit <= 0:
        raise ValueError("timelimit must be positive")
    if maxiter > 1e6:
        print("Warning: maxiter > 1e6, setting to 1e6")
        maxiter = int(1e6)
    
    # Initial scaling if H is not zero
    if np.sum(H) > 0:
        nHtH = np.linalg.norm(H.T @ H, 'fro')**2
        HtAHt = np.sum((H.T @ A) * H.T)
        scaling = HtAHt / nHtH if nHtH > 0 else 1.0
        H = np.sqrt(scaling) * H
    
    # Initial objective function
    nA = np.linalg.norm(A, 'fro')**2
    nHtH = np.linalg.norm(H.T @ H, 'fro')**2
    HtAHt = np.sum((H.T @ A) * H.T)
    e0 = 0.5 * (nA - 2*HtAHt + nHtH)
    
    if display == 'on':
        print(f'Factorizing a {n}x{n} matrix using r={r} (maxiter={maxiter}, timelimit={timelimit})')
        print(f'Initial matrix: {init_matrix}({n},{r})')
        if shuffle_columns:
            print('The columns are shuffled')
        print(f'Initial objective function={e0:.5g}')
    
    # Initialize arrays for tracking progress
    errors = [e0]
    times = [0.0]
    
    # Main coordinate descent algorithm
    start_time = time.time()
    
    # Handle sparse matrices
    is_sparse = issparse(A)
    if is_sparse:
        A = csc_matrix(A)
        A_diag = A.diagonal()
    
    for iteration in range(maxiter):
        iter_start_time = time.time()
        f_decrease = 0.0
        
        # Precompute norms and gram matrix
        nL, nC, HH = precompute_norms(H)
        
        # Column order (shuffle if requested)
        col_order = np.arange(r)
        if shuffle_columns:
            np.random.shuffle(col_order)
        
        # Iterate over columns
        for kk in range(r):
            k = col_order[kk]
            
            if is_sparse:
                # Sparse matrix version
                AH = A @ H[:, k]
            else:
                # Dense matrix version  
                AH = A @ H[:, k]
            
            # Iterate over variables in column k
            for i in range(n):
                # Compute HHH[i] = sum_j H[i,j] * (H^T H)[j,k]
                HHH_i = 0.0
                for j in range(r):
                    if j <= k:
                        HHH_i += HH[j, k] * H[i, j]
                    else:
                        HHH_i += HH[k, j] * H[i, j]
                
                h_old = H[i, k]
                
                # Compute coefficients for the polynomial optimization
                if is_sparse:
                    a = nC[k] + nL[i] - A_diag[i] - 2 * h_old**2
                else:
                    a = nC[k] + nL[i] - A[i, i] - 2 * h_old**2
                
                b = HHH_i - AH[i] - h_old * a - h_old**3
                
                # Solve the polynomial optimization problem
                h_new = best_polynomial_root(a, b)
                s1 = h_old - h_new
                
                # Update if there's a change
                if s1 != 0:
                    H[i, k] = h_new
                    s2 = h_new**2 - h_old**2
                    
                    # Update norms
                    nC[k] += s2
                    nL[i] += s2
                    
                    # Update AH for remaining variables
                    if is_sparse:
                        # For sparse matrices, update AH more carefully
                        for j in range(i, n):
                            if A[i, j] != 0:
                                AH[j] -= A[i, j] * s1
                    else:
                        # Dense matrix update
                        AH[i:] -= A[i, i:] * s1
                    
                    # Update gram matrix HH
                    for j in range(k):
                        HH[j, k] -= s1 * H[i, j]
                    HH[k, k] += s2
                    for j in range(k+1, r):
                        HH[k, j] -= s1 * H[i, j]
                    
                    # Track objective function decrease
                    f_decrease += 4*b*s1 - 2*a*s2 + h_old**4 - h_new**4
        
        # Record progress
        current_time = time.time() - start_time
        current_error = errors[-1] - 0.5 * f_decrease
        errors.append(current_error)
        times.append(current_time)
        
        # Check stopping conditions
        if current_time >= timelimit:
            if display == 'on':
                print(f"Stopped due to time limit at iteration {iteration + 1}")
            break
    
    # Final objective function
    if display == 'on':
        nHtH = np.linalg.norm(H.T @ H, 'fro')**2
        HtAHt = np.sum((H.T @ A) * H.T)
        ef = 0.5 * (nA - 2*HtAHt + nHtH)
        print(f'Final objective function={ef:.5g}')
    
    return H, np.array(errors), np.array(times)


def example_usage():
    """Example usage of the symmetric NMF algorithm."""
    # Generate a random symmetric nonnegative matrix
    n = 100
    np.random.seed(42)
    A_temp = np.random.rand(n, n)
    A = A_temp + A_temp.T  # Make symmetric
    
    # Factorization rank
    r = 10
    
    # Options
    options = {
        'maxiter': 100,
        'timelimit': 5.0,
        'init_matrix': 'random',
        'seed': 0,
        'display': 'on'
    }
    
    # Run symmetric NMF
    print("Running symmetric NMF with cyclic updates...")
    options['shuffle_columns'] = False
    H1, e1, t1 = symmetric_nmf(A, r, options)
    
    print("\nRunning symmetric NMF with shuffled updates...")
    options['shuffle_columns'] = True
    H2, e2, t2 = symmetric_nmf(A, r, options)
    
    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(t1, e1, label='CD-Cyclic', linewidth=2)
        plt.plot(t2, e2, 'r-', label='CD-Shuffle', linewidth=2)
        plt.xlabel('Time (s.)')
        plt.ylabel('Error - 1/2 * ||A - HH^T||_F^2')
        plt.legend()
        plt.title('Symmetric NMF Convergence')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")
    
    print(f"\nFinal errors:")
    print(f"Cyclic: {e1[-1]:.6f}")
    print(f"Shuffle: {e2[-1]:.6f}")
    recontructed_A1 = H1 @ H1.T

    return H1, H2, e1, e2, t1, t2


if __name__ == "__main__":
    example_usage()
