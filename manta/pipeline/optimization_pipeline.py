"""
Topic count optimization pipeline for MANTA.

This module provides functionality to find the optimal number of topics
by evaluating coherence scores across a range of topic counts.
"""

import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sparse

from .._functions.nmf import run_nmf
from ..utils.analysis.gensim_coherence import calculate_gensim_cv_coherence
from ..utils.console.console_manager import ConsoleManager, get_console


@dataclass
class OptimizationResult:
    """Container for optimization results.

    Attributes:
        topic_counts: List of topic counts that were evaluated.
        coherence_scores: List of C_V coherence scores for each topic count.
        optimal_topic_count: Topic count with highest coherence score.
        elbow_topic_count: Topic count identified by elbow detection (may be None).
        optimal_coherence: Highest coherence score achieved.
        nmf_method: NMF algorithm variant used.
        iteration_times: Time in seconds for each iteration.
        total_time: Total optimization time in seconds.
        preprocessing_time: Time spent on preprocessing in seconds.
    """

    topic_counts: List[int]
    coherence_scores: List[float]
    optimal_topic_count: int
    elbow_topic_count: Optional[int]
    optimal_coherence: float
    nmf_method: str
    iteration_times: List[float]
    total_time: float
    preprocessing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topic_counts": self.topic_counts,
            "coherence_scores": self.coherence_scores,
            "optimal_topic_count": self.optimal_topic_count,
            "elbow_topic_count": self.elbow_topic_count,
            "optimal_coherence": self.optimal_coherence,
            "nmf_method": self.nmf_method,
            "iteration_times": self.iteration_times,
            "total_time": self.total_time,
            "preprocessing_time": self.preprocessing_time,
            "average_iteration_time": (
                sum(self.iteration_times) / len(self.iteration_times)
                if self.iteration_times
                else 0
            ),
        }


class OptimizationPipeline:
    """Handles topic count optimization via coherence score evaluation."""

    @staticmethod
    def run_optimization(
        tfidf_matrix: sparse.csr_matrix,
        vocab: List[str],
        text_array: List[str],
        topic_counts: List[int],
        nmf_method: str,
        words_per_topic: int,
        lambda_val: float,
        console: Optional[ConsoleManager] = None,
    ) -> OptimizationResult:
        """
        Run NMF optimization across multiple topic counts.

        Args:
            tfidf_matrix: Preprocessed TF-IDF sparse matrix.
            vocab: Vocabulary list.
            text_array: Original text documents.
            topic_counts: List of topic counts to evaluate.
            nmf_method: NMF algorithm variant ('nmf', 'pnmf', 'nmtf').
            words_per_topic: Number of top words per topic for coherence.
            lambda_val: Lambda value for relevance scoring.
            console: Console manager for output.

        Returns:
            OptimizationResult with all evaluation metrics.
        """
        _console = console or get_console()

        coherence_scores = []
        iteration_times = []
        total_iterations = len(topic_counts)

        _console.print_header("Topic Count Optimization Loop")

        for i, topic_count in enumerate(topic_counts, 1):
            _console.print_status(
                f"Iteration {i}/{total_iterations}: Testing {topic_count} topics...",
                "processing",
            )

            iteration_start = time.time()

            # Run NMF and calculate coherence
            coherence = OptimizationPipeline._evaluate_topic_count(
                tfidf_matrix=tfidf_matrix,
                vocab=vocab,
                text_array=text_array,
                topic_count=topic_count,
                nmf_method=nmf_method,
                words_per_topic=words_per_topic,
                lambda_val=lambda_val,
                console=_console,
            )

            iteration_time = time.time() - iteration_start

            coherence_scores.append(coherence)
            iteration_times.append(iteration_time)

            _console.print_status(
                f"Topics: {topic_count} | Coherence: {coherence:.4f} | Time: {iteration_time:.2f}s",
                "success",
            )

            # Clear memory after each iteration
            gc.collect()

        # Find optimal (highest coherence)
        optimal_idx = int(np.argmax(coherence_scores))
        optimal_topic_count = topic_counts[optimal_idx]
        optimal_coherence = coherence_scores[optimal_idx]

        # Detect elbow point
        elbow_topic_count = OptimizationPipeline._detect_elbow(
            topic_counts, coherence_scores
        )

        if elbow_topic_count is not None:
            _console.print_status(
                f"Elbow detected at {elbow_topic_count} topics", "info"
            )

        return OptimizationResult(
            topic_counts=topic_counts,
            coherence_scores=coherence_scores,
            optimal_topic_count=optimal_topic_count,
            elbow_topic_count=elbow_topic_count,
            optimal_coherence=optimal_coherence,
            nmf_method=nmf_method,
            iteration_times=iteration_times,
            total_time=sum(iteration_times),
            preprocessing_time=0,  # Set by caller
        )

    @staticmethod
    def _evaluate_topic_count(
        tfidf_matrix: sparse.csr_matrix,
        vocab: List[str],
        text_array: List[str],
        topic_count: int,
        nmf_method: str,
        words_per_topic: int,
        lambda_val: float,
        console: ConsoleManager,
    ) -> float:
        """
        Run NMF for a single topic count and calculate coherence.

        Args:
            tfidf_matrix: TF-IDF matrix.
            vocab: Vocabulary list.
            text_array: Original text documents.
            topic_count: Number of topics to extract.
            nmf_method: NMF algorithm variant.
            words_per_topic: Number of words per topic.
            lambda_val: Lambda for relevance scoring.
            console: Console manager.

        Returns:
            C_V coherence score for this topic count.
        """
        # Run NMF decomposition
        nmf_output = run_nmf(
            num_of_topics=topic_count,
            sparse_matrix=tfidf_matrix,
            norm_thresh=0.005,
            nmf_method=nmf_method,
        )

        # Calculate coherence using existing gensim coherence module
        coherence_results = calculate_gensim_cv_coherence(
            h_matrix=nmf_output["H"],
            w_matrix=nmf_output["W"],
            vocabulary=vocab,
            documents=text_array,
            s_matrix=nmf_output.get("S", None),
            lambda_val=lambda_val,
            top_n_words=words_per_topic,
        )

        return coherence_results["c_v_average"]

    @staticmethod
    def _detect_elbow(
        topic_counts: List[int], coherence_scores: List[float]
    ) -> Optional[int]:
        """
        Detect the elbow point in coherence scores using the second derivative method.

        The elbow point is where the rate of improvement in coherence
        starts to decrease significantly, indicating diminishing returns
        from adding more topics.

        Args:
            topic_counts: List of topic counts evaluated.
            coherence_scores: Corresponding coherence scores.

        Returns:
            Topic count at the elbow point, or None if not detectable.
        """
        if len(coherence_scores) < 3:
            return None

        scores = np.array(coherence_scores)

        # Calculate first derivative (rate of change)
        first_diff = np.diff(scores)

        # Calculate second derivative (rate of change of slope)
        second_diff = np.diff(first_diff)

        if len(second_diff) == 0:
            return None

        # The elbow is where the second derivative is maximum
        # (greatest change in slope, indicating the curve is bending)
        # We look for the point where improvement rate drops most sharply
        elbow_idx = int(np.argmax(np.abs(second_diff))) + 1

        # Ensure the index is valid
        if elbow_idx < len(topic_counts):
            return topic_counts[elbow_idx]

        return None

    @staticmethod
    def get_recommendation(result: OptimizationResult) -> Dict[str, Any]:
        """
        Generate a recommendation based on optimization results.

        Args:
            result: Optimization result object.

        Returns:
            Dictionary with recommendation and reasoning.
        """
        recommendation = {
            "recommended_topic_count": result.optimal_topic_count,
            "method": "max_coherence",
            "coherence_score": result.optimal_coherence,
            "reasoning": (
                f"Topic count {result.optimal_topic_count} achieved the highest "
                f"coherence score of {result.optimal_coherence:.4f}."
            ),
        }

        # If elbow point differs from optimal, provide additional insight
        if (
            result.elbow_topic_count is not None
            and result.elbow_topic_count != result.optimal_topic_count
        ):
            elbow_idx = result.topic_counts.index(result.elbow_topic_count)
            elbow_coherence = result.coherence_scores[elbow_idx]

            recommendation["alternative_topic_count"] = result.elbow_topic_count
            recommendation["alternative_method"] = "elbow"
            recommendation["alternative_coherence"] = elbow_coherence
            recommendation["alternative_reasoning"] = (
                f"Elbow point at {result.elbow_topic_count} topics "
                f"(coherence: {elbow_coherence:.4f}) may provide a good "
                f"balance between model complexity and coherence."
            )

        return recommendation
