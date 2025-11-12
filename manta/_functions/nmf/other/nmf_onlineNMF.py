import numpy as np

# Example dimensions
m, n, k = 100, 50, 10  # 100 samples, 50 features, 10 latent factors

# Initialize matrices
V = np.random.rand(m, n)  # Data matrix (non-negative)
W = np.random.rand(m, k)  # Basis matrix
H = np.random.rand(k, n)  # Factor matrix

# Reconstruction
V_reconstructed = W @ H
print(f"Original shape: {V.shape}, Reconstructed shape: {V_reconstructed.shape}")


# Frobenius norm squared
def frobenius_loss(V, W, H):
    reconstruction = W @ H
    return np.sum((V - reconstruction) ** 2)

loss = frobenius_loss(V, W, H)
print(f"Reconstruction loss: {loss}")

# Example: New data arrives
p = 20  # 20 new samples
U = np.random.rand(p, n)  # New data matrix

# Stack vertically to form combined data
V_prime = np.vstack([V, U])
print(f"Original V shape: {V.shape}")
print(f"New data U shape: {U.shape}")
print(f"Combined V' shape: {V_prime.shape}")

# Goal: Find W' and H' such that V' ≈ W' @ H'
# W' will have shape (m+p, k) and H' will have shape (k, n)
W_prime_target_shape = (m + p, k)
H_prime_target_shape = (k, n)
print(f"Target W' shape: {W_prime_target_shape}")
print(f"Target H' shape: {H_prime_target_shape}")

# W' is split into two blocks
# W_1' corresponds to old data V (shape: m x k)
# W_2' corresponds to new data U (shape: p x k)

def split_W_prime(W_prime, m, p):
    """Split W' into blocks for old and new data"""
    W_1_prime = W_prime[:m, :]  # First m rows
    W_2_prime = W_prime[m:, :]  # Last p rows
    return W_1_prime, W_2_prime

# Example split
W_prime_example = np.random.rand(m + p, k)
W_1_prime, W_2_prime = split_W_prime(W_prime_example, m, p)
print(f"W_1' shape: {W_1_prime.shape} (for old data)")
print(f"W_2' shape: {W_2_prime.shape} (for new data)")

# Relationship between old and new factorizations
# P is the transformation matrix (k x k)
P = np.random.rand(k, k)  # Invertible transformation matrix
P_inv = np.linalg.inv(P)

# Theoretical relationships
W_1_prime_theory = W @ P
H_prime_theory = P_inv @ H

print(f"Transformation matrix P shape: {P.shape}")
print(f"W_1' = W @ P, shapes: {W.shape} @ {P.shape} = {W_1_prime_theory.shape}")
print(f"H' = P^(-1) @ H, shapes: {P_inv.shape} @ {H.shape} = {H_prime_theory.shape}")

# Create augmented matrix by stacking H and U vertically
H_U_augmented = np.vstack([H, U])
print(f"Augmented matrix [H; U] shape: {H_U_augmented.shape}")
print(f"H shape: {H.shape}, U shape: {U.shape}")

# This augmented matrix will be factorized as W* @ H*
# where W* has shape (k+p, k) and H* has shape (k, n)
W_star_shape = (k + p, k)
H_star_shape = (k, n)
print(f"Target W* shape: {W_star_shape}")
print(f"Target H* shape: {H_star_shape}")

# W* is also split into blocks
def create_W_star_blocks(k, p):
    """Create example W* with its blocks"""
    W_star = np.random.rand(k + p, k)
    W_1_star = W_star[:k, :]  # First k rows (k x k)
    W_2_star = W_star[k:, :]  # Last p rows (p x k)
    return W_star, W_1_star, W_2_star

W_star, W_1_star, W_2_star = create_W_star_blocks(k, p)
print(f"W_1* shape: {W_1_star.shape}")
print(f"W_2* shape: {W_2_star.shape}")

# Verify the factorization relationships
H_star = np.random.rand(k, n)

# These should hold after NMF convergence
H_reconstructed = W_1_star @ H_star
U_reconstructed = W_2_star @ H_star

print(f"H reconstruction: {W_1_star.shape} @ {H_star.shape} = {H_reconstructed.shape}")
print(f"U reconstruction: {W_2_star.shape} @ {H_star.shape} = {U_reconstructed.shape}")
print(f"Should match H shape: {H.shape}")
print(f"Should match U shape: {U.shape}")


def onmf_update_step(W, H, U, W_1_star, W_2_star):
    """
    Perform one ONMF update step

    Args:
        W: Current basis matrix (m x k)
        H: Current factor matrix (k x n)
        U: New data matrix (p x n)
        W_1_star, W_2_star: Obtained from NMF of [H; U]

    Returns:
        W_prime: Updated basis matrix
        H_prime: Updated factor matrix
    """

    # Update factor matrix: H' = (W_1*)^(-1) @ H
    W_1_star_inv = np.linalg.pinv(W_1_star)  # Use pseudo-inverse for stability
    H_prime = W_1_star_inv @ H

    # Update basis matrix: W' = [W @ W_1*; W_2*]
    W_upper = W @ W_1_star  # Old data component
    W_prime = np.vstack([W_upper, W_2_star])  # Stack with new data component

    return W_prime, H_prime


# Example usage (assuming W_1_star and W_2_star are from NMF)
W_prime, H_prime = onmf_update_step(W, H, U, W_1_star, W_2_star)

print(f"Updated W' shape: {W_prime.shape} (should be {m + p} x {k})")
print(f"Updated H' shape: {H_prime.shape} (should be {k} x {n})")

# Verify reconstruction
V_prime_reconstructed = W_prime @ H_prime
print(f"Reconstruction error: {np.sum((V_prime - V_prime_reconstructed) ** 2):.6f}")


def orthogonal_nmf_update(V, W, H, alpha=0.1, max_iter=100, tol=1e-6):
    """
    Orthogonal NMF with multiplicative updates

    Args:
        V: Data matrix (m x n)
        W: Basis matrix (m x k)
        H: Factor matrix (k x n)
        alpha: Orthogonality regularization parameter
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        W, H: Updated matrices
    """

    # Create Gamma matrix (zero diagonal, ones elsewhere)
    k = H.shape[0]
    Gamma = np.ones((k, k)) - np.eye(k)

    for iteration in range(max_iter):
        # Store old values for convergence check
        W_old, H_old = W.copy(), H.copy()

        # Update W
        VH_T = V @ H.T
        WHH_T = W @ H @ H.T
        W = W * (VH_T / (WHH_T + 1e-10))  # Add small epsilon to avoid division by zero
        W = np.maximum(W, 1e-10)  # Ensure non-negativity

        # Update H
        W_T_V = W.T @ V
        W_T_W_H = W.T @ W @ H
        Gamma_H = Gamma @ H
        H = H * (W_T_V / (W_T_W_H + alpha * Gamma_H + 1e-10))
        H = np.maximum(H, 1e-10)  # Ensure non-negativity

        # Check convergence
        W_diff = np.sum((W - W_old) ** 2)
        H_diff = np.sum((H - H_old) ** 2)
        if W_diff + H_diff < tol:
            print(f"Converged at iteration {iteration}")
            break

    return W, H


# Example usage
V_example = np.random.rand(50, 30)
W_init = np.random.rand(50, 5)
H_init = np.random.rand(5, 30)

W_updated, H_updated = orthogonal_nmf_update(V_example, W_init, H_init)
print(f"Orthogonality check - H @ H.T should be close to diagonal:")
orthogonality_matrix = H_updated @ H_updated.T
print(f"Off-diagonal sum: {np.sum(orthogonality_matrix) - np.trace(orthogonality_matrix):.6f}")


class ONMF:
    """Online Nonnegative Matrix Factorization"""

    def __init__(self, n_components, alpha=0.1, max_iter=100):
        self.k = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.W = None
        self.H = None
        self.is_initialized = False

    def initialize(self, V):
        """Initialize with first batch of data"""
        m, n = V.shape

        # Random initialization
        self.W = np.random.rand(m, self.k)
        self.H = np.random.rand(self.k, n)

        # Perform initial orthogonal NMF
        self.W, self.H = orthogonal_nmf_update(V, self.W, self.H, self.alpha, self.max_iter)
        self.is_initialized = True

        print(f"ONMF initialized with W: {self.W.shape}, H: {self.H.shape}")

    def update(self, U):
        """Update with new data batch U"""
        if not self.is_initialized:
            raise ValueError("ONMF must be initialized first")

        p, n = U.shape

        # Step 1: Create augmented matrix [H; U]
        H_U_augmented = np.vstack([self.H, U])

        # Step 2: Perform NMF on augmented matrix
        # Initialize W* and H*
        W_star = np.random.rand(self.k + p, self.k)
        H_star = np.random.rand(self.k, n)

        # Solve [H; U] = W* @ H* using orthogonal NMF
        W_star, H_star = orthogonal_nmf_update(H_U_augmented, W_star, H_star,
                                               self.alpha, self.max_iter)

        # Step 3: Split W* into blocks
        W_1_star = W_star[:self.k, :]  # First k rows
        W_2_star = W_star[self.k:, :]  # Last p rows

        # Step 4: Update using ONMF rules
        W_1_star_inv = np.linalg.pinv(W_1_star)
        H_new = W_1_star_inv @ self.H

        W_upper = self.W @ W_1_star
        W_new = np.vstack([W_upper, W_2_star])

        # Update stored matrices
        self.W = W_new
        self.H = H_new

        print(f"ONMF updated with new data. W: {self.W.shape}, H: {self.H.shape}")

        return self.W, self.H

    def transform(self, V):
        """Reconstruct data using current factorization"""
        return self.W @ self.H


# Example usage
np.random.seed(42)

# Initial data
V_initial = np.random.rand(100, 50)
onmf = ONMF(n_components=10, alpha=0.1)
onmf.initialize(V_initial)

# Simulate streaming data
for t in range(3):
    # New data batch
    U_new = np.random.rand(20, 50)
    print(f"\n--- Time step {t + 1} ---")
    W_updated, H_updated = onmf.update(U_new)
    V_initial = np.vstack([V_initial, U_new])  # Update initial data with new batch
    # Check reconstruction quality
    current_data_size = W_updated.shape[0]
    print(f"Current data covers {current_data_size} samples")

v_reconstructed = onmf.transform(V_initial)
print()