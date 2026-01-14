"""
Hierarchical clustering based on rank-two nonnegative matrix factorization.

Python implementation of the algorithm described in:
N. Gillis, D. Kuang and H. Park, "Hierarchical Clustering of 
Hyperspectral Images using Rank-Two Nonnegative Matrix Factorization", 
IEEE Trans. on Geoscience and Remote Sensing 53 (4), pp. 2066-2078, 2015.

This algorithm performs hierarchical clustering by recursively splitting clusters
using rank-two NMF, k-means, or spherical k-means.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union, Any
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
import warnings


class TreeNode:
    """Node in the hierarchical clustering tree."""
    def __init__(self, node_id: int, indices: np.ndarray, parent: Optional[int] = None):
        self.id = node_id
        self.indices = indices  # Column indices in this cluster
        self.parent = parent
        self.children = []
        self.centroid = None
        self.first_sv = 0.0  # First singular value
        self.split_criterion = -1.0  # Criterion for splitting (-1 means not computed)


class HierarchicalNMF:
    """Hierarchical clustering using rank-two NMF."""
    
    def __init__(self):
        self.nodes = {}  # Dictionary of all nodes
        self.leaf_nodes = []  # Current leaf nodes (active clusters)
        self.max_node_id = 0
        self.count = 1  # Number of current clusters
    
    def fit(self, M: np.ndarray, r: int, algo: int = 1, 
            manual_split: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hierarchical clustering.
        
        Args:
            M: (m, n) data matrix representing n data points in m-dimensional space
            r: Number of clusters to generate
            algo: Algorithm for splitting clusters:
                  1 = rank-two NMF (default)
                  2 = k-means  
                  3 = spherical k-means
            manual_split: If True, allows manual cluster selection (interactive)
        
        Returns:
            IDX: (n,) cluster assignment vector
            C: (m, r) matrix of cluster centroids
        """
        if M.min() < 0:
            warnings.warn("Input matrix contains negative entries, setting them to zero")
            M = np.maximum(M, 0)
        
        m, n = M.shape
        self.M = M
        self.algo = algo
        
        # Initialize root node
        root_indices = np.arange(n)
        root_node = TreeNode(1, root_indices)
        root_node.centroid, root_node.first_sv = self._representative_vector(M)
        self.nodes[1] = root_node
        self.leaf_nodes = [1]
        self.max_node_id = 1
        self.count = 1
        
        print("Hierarchical clustering started...")
        
        # Main clustering loop
        while self.count < r:
            # Update leaf nodes by splitting them
            self._update_leaf_nodes()
            
            # Choose cluster to split
            if self.count == 1:
                # Only root node exists, split it
                split_idx = 0
            elif not manual_split:
                # Split node with maximum criterion
                split_idx = self._choose_cluster_to_split()
            else:
                # Manual selection (simplified for non-interactive use)
                split_idx = self._choose_cluster_to_split()
                print(f"Automatically splitting cluster {split_idx + 1}")
            
            # Split the chosen cluster
            self._split_cluster(split_idx)
            
            if not manual_split and self.count % 10 == 0:
                print(f"{self.count}...")
            elif not manual_split:
                print(f"{self.count}...", end="")
        
        if not manual_split:
            print("Done.")
        
        # Generate final clustering
        IDX = self._clusters_to_vector()
        C = np.column_stack([self.nodes[leaf_id].centroid for leaf_id in self.leaf_nodes])
        
        return IDX, C
    
    def _update_leaf_nodes(self):
        """Update leaf nodes by computing split criteria."""
        for leaf_id in self.leaf_nodes:
            node = self.nodes[leaf_id]
            if node.split_criterion == -1 and len(node.indices) > 1:
                # Split this cluster to compute criterion
                clusters, centroids, first_sv = self._split_cluster_data(
                    self.M[:, node.indices], self.algo)
                
                # Create child nodes
                left_child_id = self.max_node_id + 1
                right_child_id = self.max_node_id + 2
                
                left_indices = node.indices[clusters[0]]
                right_indices = node.indices[clusters[1]]
                
                left_child = TreeNode(left_child_id, left_indices, leaf_id)
                right_child = TreeNode(right_child_id, right_indices, leaf_id)
                
                left_child.centroid, left_child.first_sv = self._representative_vector(
                    self.M[:, left_indices])
                right_child.centroid, right_child.first_sv = self._representative_vector(
                    self.M[:, right_indices])
                
                self.nodes[left_child_id] = left_child
                self.nodes[right_child_id] = right_child
                node.children = [left_child_id, right_child_id]
                
                # Compute split criterion (reduction in error)
                node.split_criterion = (left_child.first_sv**2 + right_child.first_sv**2 - 
                                      node.first_sv**2)
                
                self.max_node_id += 2
    
    def _choose_cluster_to_split(self) -> int:
        """Choose which cluster to split based on maximum criterion."""
        criteria = []
        for leaf_id in self.leaf_nodes:
            criteria.append(self.nodes[leaf_id].split_criterion)
        
        return np.argmax(criteria)
    
    def _split_cluster(self, split_idx: int):
        """Split the cluster at the given index."""
        leaf_id = self.leaf_nodes[split_idx]
        node = self.nodes[leaf_id]
        
        # Add children to leaf nodes
        self.leaf_nodes.extend(node.children)
        # Remove parent from leaf nodes
        self.leaf_nodes.pop(split_idx)
        
        self.count += 1
    
    def _split_cluster_data(self, M: np.ndarray, algo: int) -> Tuple[List[np.ndarray], 
                                                                   np.ndarray, float]:
        """
        Split cluster data into two subclusters.
        
        Args:
            M: (m, n) data matrix for the cluster
            algo: Algorithm to use (1=rank2NMF, 2=kmeans, 3=spherical kmeans)
        
        Returns:
            clusters: List of two index arrays
            centroids: (m, 2) centroid matrix  
            first_sv: First singular value
        """
        m, n = M.shape
        
        if algo == 1:  # Rank-2 NMF
            U, V, first_sv = self._rank2_nmf(M)
            
            # Normalize columns of V to sum to one
            V = V / (np.sum(V, axis=0, keepdims=True) + 1e-16)
            x = V[0, :]
            
            # Compute threshold to split cluster
            threshold = self._find_threshold(x)
            
            clusters = [np.where(x >= threshold)[0], np.where(x < threshold)[0]]
            centroids = U
            
        elif algo == 2:  # K-means
            # Use SVD for initialization
            try:
                if min(m, n) > 2:
                    u, s, vt = svds(M, k=2)
                    first_sv = s[0]
                else:
                    u, s, vt = svd(M, full_matrices=False)
                    first_sv = s[0]
                    u, s, vt = u[:, :2], s[:2], vt[:2, :]
            except:
                u, s, vt = svd(M, full_matrices=False)
                first_sv = s[0]
                u, s, vt = u[:, :min(2, len(s))], s[:min(2, len(s))], vt[:min(2, len(s)), :]
                if u.shape[1] == 1:
                    u = np.column_stack([u, u])
                    s = np.array([s[0], s[0]])
                    vt = np.row_stack([vt, vt])
            
            # Initialize with SVD result - ensure correct shape
            init_centroids = u @ np.diag(s) @ vt
            if init_centroids.shape[1] != M.shape[1]:
                # If shapes don't match, use random initialization
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            else:
                kmeans = KMeans(n_clusters=2, init=init_centroids.T, n_init=1, random_state=42)
            labels = kmeans.fit_predict(M.T)
            
            clusters = [np.where(labels == 0)[0], np.where(labels == 1)[0]]
            centroids = kmeans.cluster_centers_.T
            
        elif algo == 3:  # Spherical k-means (simplified as cosine k-means)
            # Normalize data
            M_norm = M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-16)
            
            # Use SVD for initialization
            try:
                if min(m, n) > 2:
                    u, s, vt = svds(M_norm, k=2)
                    first_sv = s[0]
                else:
                    u, s, vt = svd(M_norm, full_matrices=False)
                    first_sv = s[0]
                    u, s, vt = u[:, :2], s[:2], vt[:2, :]
            except:
                u, s, vt = svd(M_norm, full_matrices=False)
                first_sv = s[0]
                u, s, vt = u[:, :min(2, len(s))], s[:min(2, len(s))], vt[:min(2, len(s)), :]
                if u.shape[1] == 1:
                    u = np.column_stack([u, u])
            
            init_centroids = u[:, :2] @ np.diag(s[:2]) @ vt[:2, :]
            init_centroids = init_centroids / (np.linalg.norm(init_centroids, axis=0, keepdims=True) + 1e-16)
            
            # Simple spherical k-means implementation
            labels = self._spherical_kmeans(M_norm, init_centroids)
            
            clusters = [np.where(labels == 0)[0], np.where(labels == 1)[0]]
            
            # Compute centroids from original data
            if len(clusters[0]) > 0 and len(clusters[1]) > 0:
                c1 = np.mean(M[:, clusters[0]], axis=1, keepdims=True)
                c2 = np.mean(M[:, clusters[1]], axis=1, keepdims=True)
                centroids = np.column_stack([c1.flatten(), c2.flatten()])
            else:
                centroids = init_centroids
        
        # Ensure we have two non-empty clusters
        if len(clusters[0]) == 0:
            clusters[0] = np.array([0])
            clusters[1] = np.array(list(range(1, n)))
        elif len(clusters[1]) == 0:
            clusters[1] = np.array([n-1])
            clusters[0] = np.array(list(range(n-1)))
        else:
            clusters[0] = np.array(clusters[0])
            clusters[1] = np.array(clusters[1])
        
        return clusters, centroids, first_sv
    
    def _rank2_nmf(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute rank-2 NMF of matrix M."""
        m, n = M.shape
        
        # Best rank-2 approximation using SVD
        if min(m, n) == 1:
            u, s, vt = svd(M, full_matrices=False)
            U = np.abs(u[:, :1])
            V = np.abs(vt[:1, :])
            first_sv = s[0]
        else:
            try:
                if min(m, n) > 2:
                    u, s, vt = svds(M, k=2)
                    # Sort by singular values (descending)
                    idx = np.argsort(s)[::-1]
                    u, s, vt = u[:, idx], s[idx], vt[idx, :]
                else:
                    u, s, vt = svd(M, full_matrices=False)
                    u, s, vt = u[:, :2], s[:2], vt[:2, :]
            except:
                u, s, vt = svd(M, full_matrices=False)
                u, s, vt = u[:, :min(2, len(s))], s[:min(2, len(s))], vt[:min(2, len(s)), :]
            
            first_sv = s[0]
            
            # Use SPA to extract columns (simplified version)
            projected_data = vt.T * s  # Shape: (n, 2)
            K = self._spa_extract(projected_data, 2)
            
            U = np.zeros((m, 2))
            if len(K) >= 1:
                temp = u @ np.diag(s) @ vt[:, K[0]:K[0]+1]
                U[:, 0] = np.maximum(temp.flatten(), 0)
            if len(K) >= 2:
                temp = u @ np.diag(s) @ vt[:, K[1]:K[1]+1]
                U[:, 1] = np.maximum(temp.flatten(), 0)
            
            # Compute optimal V using NNLS
            V = self._anls_rank2(U.T @ U, M.T @ U)
        
        return U, V, first_sv
    
    def _spa_extract(self, X: np.ndarray, r: int) -> List[int]:
        """Simplified SPA for extracting r columns."""
        m, n = X.shape
        if r >= n:
            return list(range(n))
        
        indices = []
        remaining = set(range(n))
        
        for _ in range(min(r, n)):
            if not remaining:
                break
            
            # Find column with maximum norm
            norms = []
            remaining_list = list(remaining)
            for idx in remaining_list:
                norms.append(np.linalg.norm(X[:, idx]))
            
            best_idx = remaining_list[np.argmax(norms)]
            indices.append(best_idx)
            remaining.remove(best_idx)
            
            if len(remaining) == 0:
                break
            
            # Project remaining columns
            u = X[:, best_idx]
            u = u / (np.linalg.norm(u) + 1e-16)
            
            for idx in remaining:
                X[:, idx] = X[:, idx] - u * (u.T @ X[:, idx])
        
        return indices
    
    def _anls_rank2(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Solve NNLS problem for rank-2 case."""
        if left.shape[0] == 1:
            return np.maximum(0, right / left[0, 0])
        
        # Solve using least squares
        H = np.linalg.solve(left, right.T).T
        
        # Handle negative entries
        negative_mask = np.any(H < 0, axis=1)
        if np.any(negative_mask):
            # Simple projection for negative entries
            H[negative_mask, 0] = np.maximum(0, right[negative_mask, 0] / left[0, 0])
            H[negative_mask, 1] = np.maximum(0, right[negative_mask, 1] / left[1, 1])
            
            # Choose better solution
            for i in np.where(negative_mask)[0]:
                h1 = np.array([H[i, 0], 0])
                h2 = np.array([0, H[i, 1]])
                
                if np.linalg.norm(left @ h1 - right[i]) < np.linalg.norm(left @ h2 - right[i]):
                    H[i] = h1
                else:
                    H[i] = h2
        
        return H.T
    
    def _find_threshold(self, x: np.ndarray, s: float = 0.01) -> float:
        """Find optimal threshold for splitting based on x values."""
        # Normalize x to [0, 1]
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-16:
            return x_min
        
        x_norm = (x - x_min) / (x_max - x_min)
        
        # Grid search for optimal threshold
        delta_values = np.arange(0, 1 + s, s)
        n = len(x_norm)
        gs = 0.05
        
        f_obj = []
        
        for delta in delta_values:
            # Fraction of values <= delta
            f_del = np.sum(x_norm <= delta) / n
            
            # Number of points in interval around delta
            delta_high = min(1.0, delta + gs)
            delta_low = max(0.0, delta - gs)
            f_inter = (np.sum(x_norm <= delta_high) - np.sum(x_norm < delta_low)) / n / (delta_high - delta_low)
            
            # Objective function
            if f_del > 0 and f_del < 1:
                obj = -np.log(f_del * (1 - f_del)) + np.exp(f_inter)
            else:
                obj = np.inf
            
            f_obj.append(obj)
        
        # Find minimum
        min_idx = np.argmin(f_obj)
        threshold_norm = delta_values[min_idx]
        
        # Convert back to original scale
        threshold = x_min + threshold_norm * (x_max - x_min)
        
        return threshold
    
    def _spherical_kmeans(self, M_norm: np.ndarray, init_centroids: np.ndarray, 
                         max_iter: int = 100) -> np.ndarray:
        """Simple spherical k-means implementation."""
        centroids = init_centroids.copy()
        centroids = centroids / (np.linalg.norm(centroids, axis=0, keepdims=True) + 1e-16)
        
        for _ in range(max_iter):
            # Assign points to clusters based on cosine similarity
            similarities = M_norm.T @ centroids
            labels = np.argmax(similarities, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(2):
                cluster_points = M_norm[:, labels == k]
                if cluster_points.shape[1] > 0:
                    new_centroid = np.mean(cluster_points, axis=1)
                    new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-16)
                    new_centroids[:, k] = new_centroid
                else:
                    new_centroids[:, k] = centroids[:, k]
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return labels
    
    def _representative_vector(self, M: np.ndarray) -> Tuple[np.ndarray, float]:
        """Find most representative column from matrix M."""
        if M.shape[1] == 1:
            return M[:, 0], np.linalg.norm(M, 'fro')
        
        if M.shape[1] == 0:
            return np.zeros(M.shape[0]), 0.0
        
        # Compute SVD
        try:
            u, s, vt = svd(M, full_matrices=False)
            if len(s) > 0:
                u = np.abs(u[:, 0])
                first_sv = s[0]
            else:
                return M[:, 0], 0.0
        except:
            return M[:, 0], np.linalg.norm(M, 'fro')
        
        # Find column that best approximates first singular vector
        u_centered = u - np.mean(u)
        M_centered = M - np.mean(M, axis=0, keepdims=True)
        
        # Compute cosine similarities
        u_norm = np.linalg.norm(u_centered)
        M_norms = np.linalg.norm(M_centered, axis=0)
        
        if u_norm > 1e-16 and np.any(M_norms > 1e-16):
            cosines = (M_centered.T @ u_centered) / (u_norm * (M_norms + 1e-16))
            cosines = np.clip(cosines, -1, 1)
            # Handle NaN values
            cosines = np.nan_to_num(cosines, nan=0.0)
            errors = np.arccos(np.abs(cosines))
            best_idx = np.argmin(errors)
        else:
            # Fallback: choose column with maximum norm
            best_idx = np.argmax(np.linalg.norm(M, axis=0))
        
        return M[:, best_idx], first_sv
    
    def _clusters_to_vector(self) -> np.ndarray:
        """Convert cluster assignments to vector format."""
        n = self.M.shape[1]
        IDX = np.zeros(n, dtype=int)
        
        for cluster_id, leaf_id in enumerate(self.leaf_nodes, 1):
            node = self.nodes[leaf_id]
            IDX[node.indices] = cluster_id
        
        return IDX


def hierarchical_nmf(M: np.ndarray, r: int, algo: int = 1, 
                    manual_split: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical clustering using rank-two NMF.
    
    Args:
        M: (m, n) data matrix representing n data points in m-dimensional space
           Can also be (H, L, m) tensor which will be reshaped to (m, H*L)
        r: Number of clusters to generate
        algo: Algorithm for splitting clusters:
              1 = rank-two NMF (default)
              2 = k-means
              3 = spherical k-means
        manual_split: If True, allows manual cluster selection (simplified for batch use)
    
    Returns:
        IDX: (n,) cluster assignment vector with values in {1, 2, ..., r}
        C: (m, r) matrix where each column is a cluster centroid
    """
    # Handle tensor input
    if M.ndim == 3:
        H, L, m = M.shape
        n = H * L
        M_reshaped = np.zeros((m, n))
        for i in range(m):
            M_reshaped[i, :] = M[:, :, i].flatten()
        M = M_reshaped
    
    # Create and run hierarchical clustering
    hnmf = HierarchicalNMF()
    IDX, C = hnmf.fit(M, r, algo, manual_split)
    
    return IDX, C


def example_usage():
    """Example usage of hierarchical NMF."""
    # Generate synthetic data
    np.random.seed(42)
    
    # Create data with 4 natural clusters
    n_clusters_true = 4
    n_points_per_cluster = 50
    n_features = 10
    
    data_clusters = []
    for i in range(n_clusters_true):
        # Create cluster center
        center = np.random.rand(n_features) * 5 + i * 2
        # Generate points around center
        cluster_data = center[:, np.newaxis] + np.random.randn(n_features, n_points_per_cluster) * 0.5
        cluster_data = np.maximum(cluster_data, 0)  # Ensure non-negativity
        data_clusters.append(cluster_data)
    
    M = np.column_stack(data_clusters)
    m, n = M.shape
    
    print(f"Generated data matrix: {m} x {n}")
    print(f"True number of clusters: {n_clusters_true}")
    
    # Test different algorithms
    algorithms = {1: "Rank-2 NMF", 2: "K-means", 3: "Spherical K-means"}
    
    for algo_id, algo_name in algorithms.items():
        print(f"\n--- Testing {algo_name} ---")
        
        try:
            IDX, C = hierarchical_nmf(M, r=n_clusters_true, algo=algo_id)
            
            print(f"Generated {len(np.unique(IDX))} clusters")
            print(f"Cluster sizes: {[np.sum(IDX == i) for i in np.unique(IDX)]}")
            
            # Simple evaluation: compute within-cluster sum of squares
            wcss = 0
            for i in range(1, n_clusters_true + 1):
                cluster_data = M[:, IDX == i]
                if cluster_data.shape[1] > 0:
                    centroid = C[:, i-1:i]
                    wcss += np.sum((cluster_data - centroid)**2)
            
            print(f"Within-cluster sum of squares: {wcss:.2f}")
            
        except Exception as e:
            print(f"Error with {algo_name}: {e}")
    
    # Plot results if possible
    try:
        if m >= 2:
            plt.figure(figsize=(15, 5))
            
            # Plot original data
            plt.subplot(1, 3, 1)
            colors = plt.cm.Set1(np.linspace(0, 1, n_clusters_true))
            for i in range(n_clusters_true):
                start_idx = i * n_points_per_cluster
                end_idx = (i + 1) * n_points_per_cluster
                plt.scatter(M[0, start_idx:end_idx], M[1, start_idx:end_idx], 
                           c=[colors[i]], label=f'True Cluster {i+1}', alpha=0.7)
            plt.title('True Clusters')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            
            # Plot hierarchical NMF results
            IDX, C = hierarchical_nmf(M, r=n_clusters_true, algo=1)
            plt.subplot(1, 3, 2)
            for i in range(1, n_clusters_true + 1):
                mask = IDX == i
                if np.any(mask):
                    plt.scatter(M[0, mask], M[1, mask], 
                               c=[colors[i-1]], label=f'HierNMF Cluster {i}', alpha=0.7)
            plt.title('Hierarchical NMF Results')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            
            # Plot centroids
            plt.subplot(1, 3, 3)
            plt.scatter(C[0, :], C[1, :], c='red', s=100, marker='x', linewidth=3)
            for i, (x, y) in enumerate(zip(C[0, :], C[1, :])):
                plt.annotate(f'C{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
            plt.title('Cluster Centroids')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            
            plt.tight_layout()
            #plt.show()
            
    except ImportError:
        print("Matplotlib not available for plotting")
    
    return IDX, C


if __name__ == "__main__":
    example_usage()