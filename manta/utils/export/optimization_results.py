"""
Export utilities for optimization results.

This module provides functions to save topic count optimization results
to CSV and JSON formats for further analysis or reporting.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

if TYPE_CHECKING:
    from ...pipeline.optimization_pipeline import OptimizationResult


def save_optimization_results(
    result: "OptimizationResult",
    output_dir: str,
    save_csv: bool = True,
    save_json: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Save optimization results to CSV and/or JSON files.

    Args:
        result: OptimizationResult object containing optimization data.
        output_dir: Output directory path.
        save_csv: Whether to save results as CSV.
        save_json: Whether to save results as JSON.

    Returns:
        Dictionary with paths to saved files ('csv' and 'json' keys).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: Dict[str, Optional[str]] = {"csv": None, "json": None}

    if save_csv:
        csv_path = output_path / f"optimization_results_{result.nmf_method}.csv"
        df = pd.DataFrame(
            {
                "Topic_Count": result.topic_counts,
                "C_V_Coherence": result.coherence_scores,
                "Iteration_Time_Seconds": result.iteration_times,
            }
        )

        # Add marker columns for optimal and elbow points
        df["Is_Optimal"] = df["Topic_Count"] == result.optimal_topic_count
        if result.elbow_topic_count is not None:
            df["Is_Elbow"] = df["Topic_Count"] == result.elbow_topic_count
        else:
            df["Is_Elbow"] = False

        df.to_csv(csv_path, index=False)
        saved_files["csv"] = str(csv_path)

    if save_json:
        json_path = output_path / f"optimization_results_{result.nmf_method}.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=4)
        saved_files["json"] = str(json_path)

    return saved_files


def load_optimization_results(filepath: str) -> Dict:
    """
    Load optimization results from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Dictionary with optimization results.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(path, "r") as f:
        return json.load(f)


def generate_optimization_summary(
    result: "OptimizationResult",
) -> str:
    """
    Generate a human-readable summary of optimization results.

    Args:
        result: OptimizationResult object.

    Returns:
        Formatted string summary.
    """
    lines = [
        "=" * 60,
        "TOPIC COUNT OPTIMIZATION RESULTS",
        "=" * 60,
        "",
        f"NMF Method: {result.nmf_method.upper()}",
        f"Topic Range: {min(result.topic_counts)} to {max(result.topic_counts)}",
        f"Topics Evaluated: {len(result.topic_counts)}",
        "",
        "-" * 40,
        "RECOMMENDATIONS",
        "-" * 40,
        "",
        f"Optimal Topic Count: {result.optimal_topic_count}",
        f"Coherence Score: {result.optimal_coherence:.4f}",
        "",
    ]

    if result.elbow_topic_count is not None:
        elbow_idx = result.topic_counts.index(result.elbow_topic_count)
        elbow_coherence = result.coherence_scores[elbow_idx]
        lines.extend(
            [
                f"Elbow Point: {result.elbow_topic_count} topics",
                f"Elbow Coherence: {elbow_coherence:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "-" * 40,
            "TIMING",
            "-" * 40,
            "",
            f"Preprocessing Time: {result.preprocessing_time:.2f}s",
            f"Optimization Time: {result.total_time:.2f}s",
            f"Average per Topic Count: {result.total_time / len(result.topic_counts):.2f}s",
            "",
            "=" * 60,
        ]
    )

    return "\n".join(lines)
