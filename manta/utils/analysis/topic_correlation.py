import numpy as np

def build_correlation_graph(H: np.ndarray):
    """
    Build topic correlation graph and compute Laplacian.

    Topics are connected if their correlation exceeds threshold.
    """
    # Compute topic correlations from current H
    # Normalize H rows to unit vectors
    correlation_threshold = 0.03  # Correlation threshold for edges
    verbose = True
    epsilon = 1e-10  # To avoid division by zero
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + epsilon)

    # Compute cosine similarity (correlation)
    topic_correlation_ = H_norm @ H_norm.T

    # Create adjacency matrix (only keep high correlations)
    A = np.where(
        topic_correlation_ >= correlation_threshold,
        topic_correlation_,
        0
    )

    # Zero out diagonal

    # Compute degree matrix
    D = np.diag(np.sum(A, axis=1))

    # Graph Laplacian: L = D - A
    L_ = D - A

    if verbose:
        n_edges = np.sum(A > 0) / 2  # Divide by 2 (undirected graph)

        print(f"Topic Correlation Graph: {A.shape[0]} nodes, {n_edges} edges")

    return A, L_


if __name__ == "__main__":
    # load H matrix from npz
    H = np.load(
        "/results/radiology_imaging_MAKALE/TopicAnalysis/Output/radiology_imaging_pnmf_bpe_15/radiology_imaging_pnmf_bpe_15_model_components.npz")["H"]
    A, L = build_correlation_graph(H)
    print(A)
    print(L)