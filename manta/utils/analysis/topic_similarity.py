"""
Topic Similarity Scoring using Hybrid Weighted TF-IDF.

This module implements topic similarity computation for NMF-generated topics
by combining NMF topic weights (as TF) with IDF values from the original corpus.
"""

import numpy as np
from scipy.sparse import issparse
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List, Tuple, Optional, Union
import warnings


class HybridTFIDFTopicSimilarity:
    """
    Compute topic similarities using a hybrid approach:
    - Use NMF topic-word weights (H matrix) as TF component
    - Apply IDF values from the original corpus
    - Calculate pairwise similarities between topics

    This helps identify redundant topics and assess topic diversity.
    """

    def __init__(
        self,
        H_matrix: np.ndarray,
        vocabulary: Dict[str, int],
        idf_values: Optional[np.ndarray] = None,
        tfidf_matrix: Optional[Union[np.ndarray, 'scipy.sparse.spmatrix']] = None,
        topic_names: Optional[List[str]] = None
    ):
        """
        Initialize the topic similarity scorer.

        Parameters
        ----------
        H_matrix : np.ndarray
            Topic-word matrix from NMF factorization (n_topics x n_words).
            Each row represents a topic's distribution over words.
        vocabulary : Dict[str, int]
            Mapping from words to indices in the vocabulary.
        idf_values : Optional[np.ndarray]
            Pre-computed IDF values for each word. If None, will be computed
            from tfidf_matrix.
        tfidf_matrix : Optional[Union[np.ndarray, scipy.sparse.spmatrix]]
            Original TF-IDF matrix (n_docs x n_words) used to compute IDF values
            if idf_values is not provided.
        topic_names : Optional[List[str]]
            Names for topics (e.g., ["Topic 01", "Topic 02", ...]).
            If None, will generate names like "Topic_0", "Topic_1", etc.
        """
        self.H_matrix = H_matrix
        self.vocabulary = vocabulary
        self.n_topics, self.n_words = H_matrix.shape

        # Reverse vocabulary mapping (index to word)
        self.idx_to_word = {idx: word for word, idx in vocabulary.items()}

        # Set or generate topic names
        if topic_names is None:
            self.topic_names = [f"Topic_{i}" for i in range(self.n_topics)]
        else:
            assert len(topic_names) == self.n_topics, \
                f"Number of topic names ({len(topic_names)}) must match number of topics ({self.n_topics})"
            self.topic_names = topic_names

        # Compute or set IDF values
        if idf_values is not None:
            self.idf_values = idf_values
        elif tfidf_matrix is not None:
            self.idf_values = self._compute_idf_from_tfidf(tfidf_matrix)
        else:
            # If neither provided, use uniform IDF (all 1s) - falls back to using H matrix directly
            warnings.warn(
                "Neither idf_values nor tfidf_matrix provided. Using uniform IDF values. "
                "This essentially uses the H matrix directly without IDF weighting.",
                UserWarning
            )
            self.idf_values = np.ones(self.n_words)

        # Will store computed weighted TF-IDF vectors
        self.weighted_tfidf_vectors = None
        self.similarity_matrix = None

    def _compute_idf_from_tfidf(self, tfidf_matrix: Union[np.ndarray, 'scipy.sparse.spmatrix']) -> np.ndarray:
        """
        Compute IDF values from a TF-IDF matrix.

        The IDF values are computed by analyzing the maximum TF-IDF value
        per word across all documents. Since TF-IDF = TF * IDF, and TF <= 1
        (after normalization), the max TF-IDF value per word gives us an
        approximation of the IDF value.

        Alternative: Compute IDF from document frequencies directly:
        IDF(t) = log(N / df(t)) where N is total docs, df(t) is document frequency.

        Parameters
        ----------
        tfidf_matrix : Union[np.ndarray, scipy.sparse.spmatrix]
            TF-IDF matrix of shape (n_docs, n_words).

        Returns
        -------
        np.ndarray
            IDF values for each word (shape: n_words).
        """
        # Convert to dense if sparse
        if issparse(tfidf_matrix):
            # For large matrices, compute max without converting to dense
            # Get document frequency (number of non-zero entries per column)
            doc_freq = np.array((tfidf_matrix != 0).sum(axis=0)).flatten()
            n_docs = tfidf_matrix.shape[0]

            # Compute IDF: log(N / df) + 1
            # Add smoothing to avoid division by zero
            idf_values = np.log(n_docs / (doc_freq + 1)) + 1

            # Replace infinite values with maximum finite value
            max_finite = np.max(idf_values[np.isfinite(idf_values)])
            idf_values[~np.isfinite(idf_values)] = max_finite

        else:
            # For dense matrices, compute similarly
            doc_freq = np.sum(tfidf_matrix != 0, axis=0)
            n_docs = tfidf_matrix.shape[0]
            idf_values = np.log(n_docs / (doc_freq + 1)) + 1

            # Handle edge cases
            max_finite = np.max(idf_values[np.isfinite(idf_values)])
            idf_values[~np.isfinite(idf_values)] = max_finite

        return idf_values

    def create_weighted_tfidf_vectors(
        self,
        top_n_words: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Create weighted TF-IDF vectors for each topic.

        For each topic:
        1. Use NMF topic-word weights (H matrix row) as TF component
        2. Apply IDF weighting from the original corpus
        3. Optionally restrict to top N words
        4. Normalize the resulting vector

        Parameters
        ----------
        top_n_words : Optional[int]
            If specified, only consider the top N words per topic (by weight).
            This can help focus on the most representative words. If None,
            uses all words.
        normalize : bool
            If True, L2-normalize the vectors (default: True).

        Returns
        -------
        np.ndarray
            Weighted TF-IDF vectors of shape (n_topics, n_words).
        """
        # Create weighted vectors: TF (from H) * IDF
        weighted_vectors = self.H_matrix * self.idf_values[np.newaxis, :]

        # Optionally restrict to top N words per topic
        if top_n_words is not None and top_n_words < self.n_words:
            # Create a mask for top words per topic
            mask = np.zeros_like(weighted_vectors, dtype=bool)

            for i in range(self.n_topics):
                # Get indices of top N words for this topic
                top_indices = np.argsort(weighted_vectors[i])[-top_n_words:]
                mask[i, top_indices] = True

            # Zero out non-top words
            weighted_vectors = weighted_vectors * mask

        # Normalize vectors
        if normalize:
            # L2 normalization
            norms = np.linalg.norm(weighted_vectors, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            weighted_vectors = weighted_vectors / norms

        self.weighted_tfidf_vectors = weighted_vectors
        return weighted_vectors

    def compute_similarity_matrix(
        self,
        method: str = 'cosine',
        **kwargs
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix between topics.

        Parameters
        ----------
        method : str
            Similarity metric to use:
            - 'cosine': Cosine similarity (default)
            - 'jensen_shannon': Jensen-Shannon divergence
            - 'euclidean': Euclidean distance (converted to similarity)
            - 'dot': Dot product (for normalized vectors, equivalent to cosine)
        **kwargs : dict
            Additional arguments passed to create_weighted_tfidf_vectors
            if vectors haven't been computed yet.

        Returns
        -------
        np.ndarray
            Similarity matrix of shape (n_topics, n_topics).
            Entry [i, j] is the similarity between topic i and topic j.
        """
        # Ensure weighted vectors are computed
        if self.weighted_tfidf_vectors is None:
            self.create_weighted_tfidf_vectors(**kwargs)

        vectors = self.weighted_tfidf_vectors
        similarity_matrix = np.zeros((self.n_topics, self.n_topics))

        if method == 'cosine':
            # Cosine similarity: dot product of normalized vectors
            similarity_matrix = np.dot(vectors, vectors.T)
            # Ensure diagonal is exactly 1.0
            np.fill_diagonal(similarity_matrix, 1.0)

        elif method == 'dot':
            # Dot product (assumes vectors are already normalized)
            similarity_matrix = np.dot(vectors, vectors.T)

        elif method == 'jensen_shannon':
            # Jensen-Shannon divergence
            # First, ensure vectors are probability distributions
            vectors_prob = vectors / (vectors.sum(axis=1, keepdims=True) + 1e-10)

            for i in range(self.n_topics):
                for j in range(i, self.n_topics):
                    # Compute JS divergence
                    m = 0.5 * (vectors_prob[i] + vectors_prob[j])
                    js_div = 0.5 * (entropy(vectors_prob[i], m) + entropy(vectors_prob[j], m))
                    # Convert divergence to similarity: 1 - JS (since JS is in [0, 1])
                    similarity = 1.0 - js_div
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        elif method == 'euclidean':
            # Euclidean distance converted to similarity
            for i in range(self.n_topics):
                for j in range(i, self.n_topics):
                    dist = euclidean(vectors[i], vectors[j])
                    # Convert distance to similarity: 1 / (1 + distance)
                    similarity = 1.0 / (1.0 + dist)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        else:
            raise ValueError(
                f"Unknown similarity method: {method}. "
                f"Choose from: 'cosine', 'jensen_shannon', 'euclidean', 'dot'"
            )

        self.similarity_matrix = similarity_matrix
        return similarity_matrix

    def find_redundant_topics(
        self,
        threshold: float = 0.8,
        exclude_self: bool = True
    ) -> List[Dict[str, Union[int, str, float]]]:
        """
        Identify pairs of topics with similarity above a threshold.

        These pairs may represent redundant topics that could be merged.

        Parameters
        ----------
        threshold : float
            Similarity threshold (0 to 1). Pairs with similarity >= threshold
            are considered redundant. Default: 0.8
        exclude_self : bool
            If True, exclude self-similarities (diagonal). Default: True.

        Returns
        -------
        List[Dict]
            List of redundant topic pairs, each containing:
            - 'topic_1_idx': Index of first topic
            - 'topic_1_name': Name of first topic
            - 'topic_2_idx': Index of second topic
            - 'topic_2_name': Name of second topic
            - 'similarity': Similarity score
            Sorted by similarity (highest first).
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        redundant_pairs = []

        # Iterate over upper triangle to avoid duplicates
        for i in range(self.n_topics):
            start_j = i + 1 if exclude_self else i
            for j in range(start_j, self.n_topics):
                similarity = self.similarity_matrix[i, j]
                if similarity >= threshold:
                    redundant_pairs.append({
                        'topic_1_idx': i,
                        'topic_1_name': self.topic_names[i],
                        'topic_2_idx': j,
                        'topic_2_name': self.topic_names[j],
                        'similarity': float(similarity)
                    })

        # Sort by similarity (highest first)
        redundant_pairs.sort(key=lambda x: x['similarity'], reverse=True)

        return redundant_pairs

    def suggest_topic_merging(
        self,
        threshold: float = 0.8,
        method: str = 'greedy'
    ) -> List[Dict[str, Union[List[int], List[str], float]]]:
        """
        Suggest which topics should be merged based on similarity.

        Parameters
        ----------
        threshold : float
            Similarity threshold for merging. Default: 0.8
        method : str
            Clustering method:
            - 'greedy': Greedily merge most similar pairs
            - 'hierarchical': Use hierarchical clustering

        Returns
        -------
        List[Dict]
            List of suggested topic clusters to merge, each containing:
            - 'topic_indices': List of topic indices in the cluster
            - 'topic_names': List of topic names in the cluster
            - 'avg_similarity': Average similarity within the cluster
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        if method == 'greedy':
            # Greedy approach: iteratively merge most similar pairs
            return self._greedy_merge_suggestions(threshold)
        elif method == 'hierarchical':
            # Hierarchical clustering approach
            return self._hierarchical_merge_suggestions(threshold)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'greedy' or 'hierarchical'")

    def _greedy_merge_suggestions(self, threshold: float) -> List[Dict]:
        """Greedy merging: find connected components in similarity graph."""
        # Create adjacency matrix based on threshold
        adj_matrix = (self.similarity_matrix >= threshold).astype(int)
        np.fill_diagonal(adj_matrix, 0)  # No self-loops

        # Find connected components using DFS
        visited = set()
        clusters = []

        def dfs(node, cluster):
            visited.add(node)
            cluster.append(node)
            for neighbor in range(self.n_topics):
                if adj_matrix[node, neighbor] and neighbor not in visited:
                    dfs(neighbor, cluster)

        for i in range(self.n_topics):
            if i not in visited:
                cluster = []
                dfs(i, cluster)
                # Only include clusters with more than 1 topic
                if len(cluster) > 1:
                    # Compute average similarity within cluster
                    similarities = [
                        self.similarity_matrix[i, j]
                        for i in cluster for j in cluster if i < j
                    ]
                    avg_sim = np.mean(similarities) if similarities else 0.0

                    clusters.append({
                        'topic_indices': cluster,
                        'topic_names': [self.topic_names[idx] for idx in cluster],
                        'avg_similarity': float(avg_sim)
                    })

        # Sort by average similarity (highest first)
        clusters.sort(key=lambda x: x['avg_similarity'], reverse=True)
        return clusters

    def _hierarchical_merge_suggestions(self, threshold: float) -> List[Dict]:
        """Hierarchical clustering approach."""
        # Convert similarity to distance
        distance_matrix = 1.0 - self.similarity_matrix

        # Perform hierarchical clustering
        # Use distance_threshold to cut the dendrogram
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - threshold,
            metric='precomputed',
            linkage='average'
        )

        # Fit on distance matrix
        labels = clustering.fit_predict(distance_matrix)

        # Group topics by cluster label
        clusters_dict = {}
        for topic_idx, label in enumerate(labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(topic_idx)

        # Format results
        clusters = []
        for cluster_topics in clusters_dict.values():
            if len(cluster_topics) > 1:  # Only include multi-topic clusters
                # Compute average similarity within cluster
                similarities = [
                    self.similarity_matrix[i, j]
                    for i in cluster_topics for j in cluster_topics if i < j
                ]
                avg_sim = np.mean(similarities) if similarities else 0.0

                clusters.append({
                    'topic_indices': cluster_topics,
                    'topic_names': [self.topic_names[idx] for idx in cluster_topics],
                    'avg_similarity': float(avg_sim)
                })

        # Sort by average similarity (highest first)
        clusters.sort(key=lambda x: x['avg_similarity'], reverse=True)
        return clusters

    def get_topic_clusters(
        self,
        n_clusters: Optional[int] = None,
        method: str = 'hierarchical',
        **kwargs
    ) -> Dict[int, List[int]]:
        """
        Cluster topics based on similarity.

        Parameters
        ----------
        n_clusters : Optional[int]
            Number of clusters to create. If None, uses distance_threshold
            from kwargs or defaults to n_topics // 2.
        method : str
            Clustering method (currently only 'hierarchical' supported).
        **kwargs : dict
            Additional arguments for clustering (e.g., distance_threshold).

        Returns
        -------
        Dict[int, List[int]]
            Dictionary mapping cluster ID to list of topic indices.
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # Convert similarity to distance
        distance_matrix = 1.0 - self.similarity_matrix

        if method == 'hierarchical':
            if n_clusters is None and 'distance_threshold' not in kwargs:
                n_clusters = max(2, self.n_topics // 2)

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average',
                **kwargs
            )

            labels = clustering.fit_predict(distance_matrix)

            # Group by cluster
            clusters = {}
            for topic_idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(topic_idx)

            return clusters
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def compute_topic_diversity_score(self) -> float:
        """
        Compute overall topic diversity score based on inter-topic similarity.

        Lower average similarity = higher diversity = better score.

        Returns
        -------
        float
            Diversity score in range [0, 1], where:
            - 1.0 = maximum diversity (all topics completely different)
            - 0.0 = minimum diversity (all topics identical)
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # Compute average similarity (excluding diagonal)
        mask = ~np.eye(self.n_topics, dtype=bool)
        avg_similarity = self.similarity_matrix[mask].mean()

        # Convert to diversity: 1 - average_similarity
        diversity = 1.0 - avg_similarity

        return float(diversity)

    def get_most_similar_topics(
        self,
        topic_idx: int,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[Tuple[int, str, float]]:
        """
        Find the most similar topics to a given topic.

        Parameters
        ----------
        topic_idx : int
            Index of the query topic.
        top_k : int
            Number of most similar topics to return.
        exclude_self : bool
            If True, exclude the topic itself from results.

        Returns
        -------
        List[Tuple[int, str, float]]
            List of (topic_index, topic_name, similarity) tuples,
            sorted by similarity (highest first).
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # Get similarities for this topic
        similarities = self.similarity_matrix[topic_idx].copy()

        if exclude_self:
            similarities[topic_idx] = -np.inf  # Exclude self

        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Format results
        results = [
            (int(idx), self.topic_names[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Compute summary statistics for topic similarities.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'diversity_score': Overall diversity score
            - 'avg_similarity': Average pairwise similarity (excluding diagonal)
            - 'min_similarity': Minimum similarity (excluding diagonal)
            - 'max_similarity': Maximum similarity (excluding diagonal)
            - 'std_similarity': Standard deviation of similarities
            - 'median_similarity': Median similarity
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # Get off-diagonal values
        mask = ~np.eye(self.n_topics, dtype=bool)
        off_diag = self.similarity_matrix[mask]

        return {
            'diversity_score': float(self.compute_topic_diversity_score()),
            'avg_similarity': float(off_diag.mean()),
            'min_similarity': float(off_diag.min()),
            'max_similarity': float(off_diag.max()),
            'std_similarity': float(off_diag.std()),
            'median_similarity': float(np.median(off_diag))
        }

    def to_dict(self) -> Dict:
        """
        Export all computed results to a dictionary.

        Returns
        -------
        Dict
            Dictionary containing all computed metrics and results.
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        return {
            'n_topics': self.n_topics,
            'topic_names': self.topic_names,
            'similarity_matrix': self.similarity_matrix.tolist(),
            'summary_statistics': self.get_summary_statistics(),
            'redundant_pairs': self.find_redundant_topics(threshold=0.8),
            'merge_suggestions': self.suggest_topic_merging(threshold=0.8)
        }
