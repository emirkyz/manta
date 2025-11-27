"""
Topic Count Optimization Script for MANTA

This script finds the optimal number of topics by:
1. Preprocessing data once and creating TF-IDF matrix
2. Running NMF with different topic counts
3. Calculating coherence scores for each
4. Plotting coherence vs topic count to find the optimal value

Usage:
    Configure the variables in the Configuration section below and run:
    python optimize_topic_count.py
"""

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
import scipy.sparse as sparse

# Import MANTA components
from manta._functions.common_language.emoji_processor import EmojiMap
from manta.utils.database.database_manager import DatabaseManager
from manta.utils.console.console_manager import ConsoleManager
from manta.pipeline import DataPipeline, TextPipeline
from manta._functions.nmf import run_nmf
from manta._functions.common_language.topic_extractor import topic_extract
from manta.utils.analysis.coherence_score import calculate_coherence_scores


# ========================================
# CONFIGURATION SECTION - EDIT THESE VALUES
# ========================================

# File and column settings
FILE_PATH = "veri_setleri/radiology_imaging.csv"
COLUMN_NAME = "abstract"
SEPARATOR = ","
LANGUAGE = "EN"  # "EN" or "TR"

# NMF settings
NMF_METHOD = "nmf"  # Options: "nmf", "pnmf", "nmtf"
TOPIC_RANGE = (2, 16)  # (min_topics, max_topics) - inclusive
WORDS_PER_TOPIC = 15  # Number of words to extract per topic

# Text processing settings
LEMMATIZE = True
TOKENIZER_TYPE = "bpe"  # Options: "bpe", "wordpiece"

# Output settings
OUTPUT_DIR = "topic_optimization_results"
SAVE_RESULTS_CSV = True
SAVE_PLOTS = True

# Optional: Data filtering (set filter_app=True to enable)
FILTER_APP = False
DATA_FILTER_OPTIONS = {
    "filter_app_country": "TR",
    "filter_app_country_column": "reviewer_language",
    "filter_app_name": "",
    "filter_app_column": "package_name",
}

# ========================================
# END CONFIGURATION SECTION
# ========================================


def load_and_preprocess_data(
    filepath: str,
    column: str,
    separator: str,
    language: str,
    lemmatize: bool,
    tokenizer_type: str,
    filter_app: bool,
    data_filter_options: Dict[str, Any],
    output_dir: str,
    console: ConsoleManager
) -> Tuple[sparse.csr_matrix, List[str], List[str], Dict[str, Any]]:
    """
    Load and preprocess data to create TF-IDF matrix (only done once).

    Returns:
        Tuple of (tfidf_matrix, vocabulary, text_array, options)
    """
    console.print_header("Data Loading & Preprocessing")
    console.print_status("Starting data preprocessing...", "processing")

    # Initialize database configuration
    db_config = DatabaseManager.initialize_database_config(output_dir)

    # Create options dictionary
    options = {
        "LEMMATIZE": lemmatize,
        "N_TOPICS": WORDS_PER_TOPIC,
        "DESIRED_TOPIC_COUNT": 5,  # Placeholder, will be overridden
        "tokenizer_type": tokenizer_type,
        "tokenizer": None,
        "nmf_type": NMF_METHOD,
        "LANGUAGE": language,
        "separator": separator,
        "gen_cloud": False,  # Disable visualizations for optimization
        "save_excel": False,
        "word_pairs_out": False,
        "gen_topic_distribution": False,
        "emoji_map": EmojiMap() if language == "TR" else None,
        "filter_app": filter_app,
        "data_filter_options": data_filter_options,
        "save_to_db": False
    }

    # Load data
    console.print_status("Loading data from file...", "processing")
    df = DataPipeline.load_data_file(filepath, options, console)

    # Preprocess dataframe
    console.print_status("Preprocessing dataframe...", "processing")
    df = DataPipeline.preprocess_dataframe(
        df, column, options, db_config.main_db_engine, "optimization_temp", console
    )

    # Perform text processing to get TF-IDF matrix
    console.print_status("Performing text processing and TF-IDF vectorization...", "processing")
    tdm, vocab, counterized_data, text_array, options = TextPipeline.perform_text_processing(
        df, column, options, console
    )

    console.print_status(f"TF-IDF matrix created: {tdm.shape[0]} documents × {tdm.shape[1]} features", "success")

    return tdm, vocab, text_array, options


def run_nmf_for_topic_count(
    tfidf_matrix: sparse.csr_matrix,
    topic_count: int,
    nmf_method: str,
    console: ConsoleManager
) -> Dict[str, Any]:
    """
    Run NMF decomposition for a specific topic count.

    Returns:
        Dictionary with W, H matrices (and S for NMTF)
    """
    console.print_status(f"Running {nmf_method.upper()} with {topic_count} topics...", "processing")

    nmf_output = run_nmf(
        num_of_topics=topic_count,
        sparse_matrix=tfidf_matrix,
        norm_thresh=0.005,
        nmf_method=nmf_method,
    )

    return nmf_output


def extract_topics_and_calculate_coherence(
    nmf_output: Dict[str, Any],
    topic_count: int,
    vocab: List[str],
    text_array: List[str],
    options: Dict[str, Any],
    console: ConsoleManager
) -> Tuple[Dict, float]:
    """
    Extract topic words from NMF results and calculate coherence score.

    Returns:
        Tuple of (topic_word_scores, gensim_c_v_coherence)
    """
    # Extract topics
    if options["LANGUAGE"] == "EN":
        word_result, _ = topic_extract(
            H=nmf_output["H"],
            W=nmf_output["W"],
            doc_word_pairs=nmf_output.get("S", None),
            topic_count=topic_count,
            vocab=vocab,
            documents=text_array,
            db_config=None,
            data_frame_name=None,
            word_per_topic=options["N_TOPICS"],
            include_documents=False,  # Skip document extraction for speed
            emoji_map=options["emoji_map"],
        )
    else:  # Turkish
        word_result, _ = topic_extract(
            H=nmf_output["H"],
            W=nmf_output["W"],
            doc_word_pairs=nmf_output.get("S", None),
            topic_count=topic_count,
            vocab=vocab,
            tokenizer=options["tokenizer"],
            documents=text_array,
            db_config=None,
            data_frame_name=None,
            word_per_topic=options["N_TOPICS"],
            include_documents=False,
            emoji_map=options["emoji_map"],
        )

    # Convert word_result to topic_word_scores format (dict of dicts)
    topic_word_scores = {}
    for topic_name, word_score_list in word_result.items():
        topic_scores = {}
        for word_score_str in word_score_list:
            if ':' in word_score_str:
                parts = word_score_str.rsplit(':', 1)
                word = parts[0]
                score = float(parts[1])
                topic_scores[word] = score
        topic_word_scores[topic_name] = topic_scores

    # Calculate coherence scores
    console.print_status(f"Calculating coherence for {topic_count} topics...", "processing")
    coherence_results = calculate_coherence_scores(
        topic_word_scores=topic_word_scores,
        output_dir=None,  # Don't save intermediate results
        table_name=None,
        column_name=None,
        cleaned_data=text_array,
        topic_word_matrix=nmf_output["H"],
        doc_topic_matrix=nmf_output["W"],
        vocabulary=vocab,
        tokenizer=options.get("tokenizer"),
        emoji_map=options["emoji_map"],
        s_matrix=nmf_output.get("S", None),
    )

    # Extract the Gensim C_V coherence score
    gensim_c_v = coherence_results.get("gensim", {}).get("c_v_average", 0.0)

    return topic_word_scores, gensim_c_v


def plot_coherence_results(
    topic_counts: List[int],
    coherence_scores: List[float],
    output_dir: str,
    nmf_method: str,
    save_plot: bool = True
) -> None:
    """
    Plot coherence scores vs topic counts and find optimal value.
    """
    # Find the optimal topic count (highest coherence)
    optimal_idx = np.argmax(coherence_scores)
    optimal_topics = topic_counts[optimal_idx]
    optimal_coherence = coherence_scores[optimal_idx]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(topic_counts, coherence_scores, 'b-o', linewidth=2, markersize=8, label='Coherence Score')
    plt.plot(optimal_topics, optimal_coherence, 'r*', markersize=20,
             label=f'Optimal: {optimal_topics} topics (C_V={optimal_coherence:.4f})')

    plt.xlabel('Number of Topics', fontsize=12)
    plt.ylabel('Gensim C_V Coherence Score', fontsize=12)
    plt.title(f'Topic Count Optimization - {nmf_method.upper()} Method', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_file = output_path / f"coherence_vs_topics_{nmf_method}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_file}")

    plt.show()

    print(f"\n{'='*60}")
    print(f"OPTIMAL TOPIC COUNT: {optimal_topics}")
    print(f"COHERENCE SCORE: {optimal_coherence:.4f}")
    print(f"{'='*60}\n")


def save_results_to_csv(
    topic_counts: List[int],
    coherence_scores: List[float],
    output_dir: str,
    nmf_method: str
) -> None:
    """
    Save optimization results to CSV file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({
        'Topic_Count': topic_counts,
        'C_V_Coherence': coherence_scores
    })

    # Save to CSV
    csv_file = output_path / f"optimization_results_{nmf_method}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Results saved to: {csv_file}")

    # Also save as JSON for more detailed info
    results_dict = {
        'nmf_method': nmf_method,
        'topic_counts': topic_counts,
        'coherence_scores': coherence_scores,
        'optimal_topic_count': int(topic_counts[np.argmax(coherence_scores)]),
        'optimal_coherence': float(np.max(coherence_scores))
    }

    json_file = output_path / f"optimization_results_{nmf_method}.json"
    with open(json_file, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"JSON results saved to: {json_file}")


def main():
    """
    Main optimization workflow.
    """
    # Initialize console
    console = ConsoleManager()
    console.start_timing()

    console.print_header(
        "MANTA Topic Count Optimization",
        f"Finding optimal topic count using {NMF_METHOD.upper()} method"
    )

    print(f"\nConfiguration:")
    print(f"  File: {FILE_PATH}")
    print(f"  Column: {COLUMN_NAME}")
    print(f"  Language: {LANGUAGE}")
    print(f"  NMF Method: {NMF_METHOD}")
    print(f"  Topic Range: {TOPIC_RANGE[0]} to {TOPIC_RANGE[1]}")
    print(f"  Tokenizer: {TOKENIZER_TYPE}")
    print(f"  Output Dir: {OUTPUT_DIR}\n")

    # Step 1: Load and preprocess data (done once)
    start_time = time.time()
    tfidf_matrix, vocab, text_array, options = load_and_preprocess_data(
        filepath=FILE_PATH,
        column=COLUMN_NAME,
        separator=SEPARATOR,
        language=LANGUAGE,
        lemmatize=LEMMATIZE,
        tokenizer_type=TOKENIZER_TYPE,
        filter_app=FILTER_APP,
        data_filter_options=DATA_FILTER_OPTIONS,
        output_dir=OUTPUT_DIR,
        console=console
    )
    preprocessing_time = time.time() - start_time
    console.print_status(f"Preprocessing completed in {preprocessing_time:.2f} seconds", "success")

    # Step 2: Loop through topic counts and calculate coherence
    console.print_header("Topic Count Optimization Loop")
    topic_counts = []
    coherence_scores = []

    min_topics, max_topics = TOPIC_RANGE
    total_iterations = max_topics - min_topics + 1

    for i, topic_count in enumerate(range(min_topics, max_topics + 1), 1):
        console.print_header(f"Iteration {i}/{total_iterations}: Testing {topic_count} Topics")

        iteration_start = time.time()

        # Run NMF
        nmf_output = run_nmf_for_topic_count(
            tfidf_matrix=tfidf_matrix,
            topic_count=topic_count,
            nmf_method=NMF_METHOD,
            console=console
        )

        # Extract topics and calculate coherence
        topic_word_scores, coherence = extract_topics_and_calculate_coherence(
            nmf_output=nmf_output,
            topic_count=topic_count,
            vocab=vocab,
            text_array=text_array,
            options=options,
            console=console
        )

        iteration_time = time.time() - iteration_start

        # Store results
        topic_counts.append(topic_count)
        coherence_scores.append(coherence)

        console.print_status(
            f"Topics: {topic_count} | Coherence: {coherence:.4f} | Time: {iteration_time:.2f}s",
            "success"
        )
        print()

    # Step 3: Plot and analyze results
    console.print_header("Results Analysis")

    if SAVE_PLOTS:
        plot_coherence_results(
            topic_counts=topic_counts,
            coherence_scores=coherence_scores,
            output_dir=OUTPUT_DIR,
            nmf_method=NMF_METHOD,
            save_plot=True
        )

    # Step 4: Save results
    if SAVE_RESULTS_CSV:
        save_results_to_csv(
            topic_counts=topic_counts,
            coherence_scores=coherence_scores,
            output_dir=OUTPUT_DIR,
            nmf_method=NMF_METHOD
        )

    # Print summary
    total_time = time.time() - start_time
    console.print_header("Optimization Complete")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Preprocessing time: {preprocessing_time:.2f} seconds")
    print(f"Optimization time: {total_time - preprocessing_time:.2f} seconds")
    print(f"Average time per topic count: {(total_time - preprocessing_time) / total_iterations:.2f} seconds")


if __name__ == "__main__":
    main()
