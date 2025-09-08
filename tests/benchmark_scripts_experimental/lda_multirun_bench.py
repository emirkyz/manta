#!/usr/bin/env python3
"""
LDA Performance Benchmark - Memory and timing analysis for LDA topic modeling

This benchmark runs LDA topic modeling multiple times to measure:
- Execution time consistency
- Peak memory usage
"""

import os
import sys
import time
import statistics
import tracemalloc
import gc

# Add parent directory to path to import standalone_lda
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from standalone_lda import compare_lda_with_nmf
from manta._functions.common_language.emoji_processor import EmojiMap


def run_lda_analysis():
    """Run LDA topic analysis"""
    file_path = "../../veri_setleri/mimic_train_impressions.csv"
    column = "report"

    # Extract base name from filepath for output naming
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    base_name = f"{base_name}_lda_benchmark"

    results = compare_lda_with_nmf(
        filepath=file_path,
        desired_columns=column,
        num_topics=10,
        base_name=base_name,
        lemmatize=True,
        num_words=15,
        generate_wordclouds=False,  # Disable for faster benchmarking
        separator=","
    )

    return results


def main():
    print("LDA Performance Benchmark")
    print("=" * 40)

    execution_times = []
    memory_usages = []
    coherence_scores = []

    for i in range(10):
        print(f"Run {i + 1}/10...")

        # Clean up memory before measurement
        gc.collect()

        # Start memory tracing
        tracemalloc.start()
        start_time = time.time()

        try:
            # Run the analysis
            result = run_lda_analysis()

            # Measure time and memory
            execution_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Convert to MB
            peak_mb = peak / 1024 / 1024
            coherence = result.get('coherence_score', 0.0)

            execution_times.append(execution_time)
            memory_usages.append(peak_mb)
            coherence_scores.append(coherence)

            print(f"  Time: {execution_time:.2f}s, Peak Memory: {peak_mb:.2f} MB, Coherence: {coherence:.4f}")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            tracemalloc.stop()

    if execution_times:  # Only show results if we have successful runs
        print("\n" + "=" * 40)
        print("RESULTS")
        print("=" * 40)
        print(f"Avg execution time: {statistics.mean(execution_times):.2f} ± {statistics.stdev(execution_times):.2f}s")
        print(f"Avg peak memory:    {statistics.mean(memory_usages):.2f} ± {statistics.stdev(memory_usages):.2f} MB")
        print(f"Max peak memory:    {max(memory_usages):.2f} MB")
        print(f"Avg coherence:      {statistics.mean(coherence_scores):.4f} ± {statistics.stdev(coherence_scores):.4f}")
    else:
        print("\n❌ No successful runs to analyze")


if __name__ == '__main__':
    main()