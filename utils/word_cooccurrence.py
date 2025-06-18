import os
import json
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
def calc_word_cooccurrence(H, sozluk, base_dir, table_name, top_n=100, min_score=1, language="EN",tokenizer:Tokenizer = None):
    """
    Calculates word co-occurrence matrix from NMF H matrix and saves results to a JSON file.

    This function computes how often words appear together in the same document based on 
    the NMF topic-word matrix (H). It identifies the most frequent word pairs, returns 
    them as a DataFrame, and saves them to a JSON file in the specified output directory.

    Args:
        H (numpy.ndarray): Topic-word matrix from NMF decomposition.
        sozluk (list): Vocabulary list where indices correspond to word IDs.
        base_dir (str): Base directory path for saving output.
        table_name (str): Name of the table/dataset for file naming.
        top_n (int, optional): Number of top pairs to return. Default is 100.
        min_score (float, optional): Minimum co-occurrence score to consider. Default is 1.

    Returns:
        pandas.DataFrame: DataFrame with columns 'Word 1', 'Word 2', and 'Score',
                         containing the top word pairs sorted by co-occurrence score.

    Side Effects:
        - Creates directory structure: {base_dir}/Output/{table_name}/
        - Saves JSON file: {table_name}_cooccurrence.json in the created directory
        :param H:
        :param sozluk:
        :param base_dir:
        :param top_n:
        :param min_score:
        :param language:
        :param tokenizer:
    """
    print("Calculating word co-occurrence matrix...")

    # Calculate co-occurrence matrix
    X = H.T @ H

    # Filter scores and keep only upper triangle to avoid duplicates
    top_scores = np.where(X > min_score, X, 0)
    top_scores = np.triu(top_scores, k=1)

    # Find non-zero indices
    top_indices = np.argwhere(top_scores > 0)

    # Create list of word pairs with scores
    top_pairs = []
    for i, j in top_indices:
        if i != j:
            score = top_scores[i, j]
            word_i = sozluk[i] if language == "EN" else tokenizer.id_to_token(i)
            word_j = sozluk[j] if language == "EN" else tokenizer.id_to_token(j)
            top_pairs.append((word_i, word_j, score))

    # Sort pairs by score and take top_n
    top_pairs = sorted(top_pairs, key=lambda x: x[2], reverse=True)[:top_n]


    # Prepare output directory
    output_dir = os.path.join(base_dir, "Output")
    table_output_dir = os.path.join(output_dir, table_name)
    os.makedirs(table_output_dir, exist_ok=True)

    # Convert DataFrame to dict for JSON serialization
    cooccurrence_data = {
        "pairs": [
            {"word_1": pair[0], "word_2": pair[1], "score": pair[2]}
            for pair in top_pairs
        ]
    }

    # Save to file
    cooccurrence_file = os.path.join(table_output_dir, f"{table_name}_cooccurrence.json")
    try:
        with open(cooccurrence_file, "w", encoding="utf-8") as f:
            json.dump(cooccurrence_data, f, indent=4, ensure_ascii=False)
        print(f"Word co-occurrence data saved to: {cooccurrence_file}")
    except Exception as e:
        print(f"Error saving word co-occurrence data: {e}")

    return cooccurrence_data
