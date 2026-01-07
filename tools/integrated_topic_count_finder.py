"""
Topic Count Optimization Script for MANTA

This script finds the optimal number of topics using the MANTA optimization API.
It evaluates coherence scores across a range of topic counts and recommends
both the optimal (highest coherence) and elbow point.

Usage:
    Configure the variables in the Configuration section below and run:
    uv run python find_topic_count.py
"""

from manta import run_optimization

# ========================================
# CONFIGURATION SECTION - EDIT THESE VALUES
# ========================================

# File and column settings
FILE_PATH = "../custom_datasets/beyza_hoca_veri.csv"
COLUMN_NAME = "abstract"
SEPARATOR = ","
LANGUAGE = "EN"  # "EN" or "TR"

# NMF settings
NMF_METHOD = "pnmf"  # Options: "nmf", "pnmf", "nmtf"
MIN_TOPICS = 1
MAX_TOPICS = 20
STEP = 1  # Evaluate every nth topic count
WORDS_PER_TOPIC = 15  # Number of words to extract per topic

# Text processing settings
LEMMATIZE = True
TOKENIZER_TYPE = "bpe"  # Options: "bpe", "wordpiece"
PAGERANK_COLUMN = "pagerank"  # Column name for PageRank weights (set to None to disable)
N_GRAMS_TO_DISCOVER = 1000  # int, "auto", or None to disable n-gram discovery

# Output settings
OUTPUT_DIR = "../custom_datasets/heart_failure_topic_count_find_avg_citiation_rank"
SAVE_PLOT = True
SHOW_PLOT = False  # Set to True to display interactive plot
SAVE_CSV = True
SAVE_JSON = True

# Cache settings
USE_CACHE = False  # Use cached TF-IDF matrix if available
FORCE_REPROCESS = False  # Set to True to ignore cache and reprocess

# ========================================
# END CONFIGURATION SECTION
# ========================================


def main():
    """Run topic count optimization."""
    result = run_optimization(
        filepath=FILE_PATH,
        column=COLUMN_NAME,
        separator=SEPARATOR,
        language=LANGUAGE,
        min_topics=MIN_TOPICS,
        max_topics=MAX_TOPICS,
        n_grams_to_discover=N_GRAMS_TO_DISCOVER,
        step=STEP,
        nmf_method=NMF_METHOD,
        lemmatize=LEMMATIZE,
        tokenizer_type=TOKENIZER_TYPE,
        words_per_topic=WORDS_PER_TOPIC,
        pagerank_column=PAGERANK_COLUMN,
        save_plot=SAVE_PLOT,
        show_plot=SHOW_PLOT,
        save_csv=SAVE_CSV,
        save_json=SAVE_JSON,
        output_dir=OUTPUT_DIR,
        use_cache=USE_CACHE,
        force_reprocess=FORCE_REPROCESS,
    )

    if result["state"] == "SUCCESS":
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Optimal Topic Count: {result['optimal_topic_count']}")
        print(f"Optimal Coherence: {result['optimal_coherence']:.4f}")
        if result.get("elbow_topic_count"):
            print(f"Elbow Point: {result['elbow_topic_count']} topics")
        print(f"Output Directory: {result['output_dir']}")
        print("=" * 60)
    else:
        print(f"\nOptimization failed: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
