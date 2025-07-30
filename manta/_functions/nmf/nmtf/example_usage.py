from .nmtf import nmtf
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
X = np.random.rand(400, 100)

X = sp.csc_matrix(X)


W, S, H = nmtf(X, rank_factor=1.0, norm_thresh=1.0, zero_threshold=0.0001)

print("W shape:", W.shape)
print("S shape:", S.shape)
print("H shape:", H.shape)


reconstructed_X = W @ S @ H
print("Reconstructed X shape:", reconstructed_X.shape)
X = X.tocsr()
print("Reconstruction error:", norm(X - reconstructed_X, 'fro'))