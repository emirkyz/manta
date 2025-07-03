import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer

def calc_word_cooccurrence(H, sozluk, base_dir, table_name, top_n=100, min_score=1, language="EN", tokenizer: Tokenizer = None, create_heatmap=True, heatmap_size=20):
    """
    Calculates word co-occurrence matrix from NMF H matrix and saves results to a JSON file.

    This function computes how often words appear together in the same document based on 
    the NMF topic-word matrix (H). It identifies the most frequent word pairs, returns 
    them as a DataFrame, and saves them to a JSON file in the specified output directory.
    Optionally creates a heatmap visualization of the top co-occurring words.

    Args:
        H (numpy.ndarray): Topic-word matrix from NMF decomposition.
        sozluk (list): Vocabulary list where indices correspond to word IDs.
        base_dir (str): Base directory path for saving output.
        table_name (str): Name of the table/dataset for file naming.
        top_n (int, optional): Number of top pairs to return. Default is 100.
        min_score (float, optional): Minimum co-occurrence score to consider. Default is 1.
        language (str, optional): Language code ("EN" or other). Default is "EN".
        tokenizer (Tokenizer, optional): Tokenizer for non-English languages.
        create_heatmap (bool, optional): Whether to create and save a heatmap. Default is True.
        heatmap_size (int, optional): Number of top words to include in heatmap. Default is 20.

    Returns:
        pandas.DataFrame: DataFrame with columns 'Word 1', 'Word 2', and 'Score',
                         containing the top word pairs sorted by co-occurrence score.

    Side Effects:
        - Creates directory structure: {base_dir}/Output/{table_name}/
        - Saves JSON file: {table_name}_cooccurrence.json in the created directory
        - Optionally saves heatmap: {table_name}_cooccurrence_heatmap.png
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
    # Check if base_dir already includes the table_name to avoid double nesting
    if base_dir.endswith(table_name):
        table_output_dir = base_dir
    else:
        output_dir = os.path.join(base_dir, "Output")
        table_output_dir = os.path.join(output_dir, table_name)
    os.makedirs(table_output_dir, exist_ok=True)

    #TODO: Will be fixed. it doesnt show all the numbers in the heatmap for some reason.

    # Create heatmap if requested
    if create_heatmap and top_pairs:
        print("Creating word co-occurrence heatmap...")
        
        # Get unique words from top pairs
        all_words = set()
        for pair in top_pairs[:heatmap_size * 2]:  # Get more pairs to ensure we have enough unique words
            all_words.add(pair[0])
            all_words.add(pair[1])
        
        # Limit to heatmap_size words
        unique_words = list(all_words)[:heatmap_size]
        
        # Create word to index mapping
        word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        
        # Create heatmap matrix
        heatmap_matrix = np.zeros((len(unique_words), len(unique_words)))
        
        # Fill the heatmap matrix with co-occurrence scores
        for word1, word2, score in top_pairs:
            if word1 in word_to_idx and word2 in word_to_idx:
                i, j = word_to_idx[word1], word_to_idx[word2]
                heatmap_matrix[i, j] = score
                heatmap_matrix[j, i] = score  # Make it symmetric
        
        # Create mask for upper triangle (including diagonal)
        mask = np.triu(np.ones_like(heatmap_matrix, dtype=bool))
        
        # Create the heatmap
        plt.figure(figsize=(12, 10))
        
        # Create heatmap with seaborn, masking upper triangle
        sns.heatmap(
            heatmap_matrix,
            mask=mask,
            xticklabels=unique_words,
            yticklabels=unique_words,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Co-occurrence Score'},
            square=True
        )
        
        plt.title(f'Word Co-occurrence Heatmap - {table_name}', fontsize=16, pad=20)
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Words', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save heatmap
        heatmap_file = os.path.join(table_output_dir, f"{table_name}_cooccurrence_heatmap.png")
        try:
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {heatmap_file}")
        except Exception as e:
            print(f"Error saving heatmap: {e}")
        finally:
            plt.close()  # Close the figure to free memory

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
