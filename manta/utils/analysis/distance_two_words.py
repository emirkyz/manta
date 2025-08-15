import Levenshtein
"""This module provides functions to calculate various string distance metrics between two words.

The module implements three different distance metrics:
- Levenshtein distance: Minimum number of single-character edits required to change one word into another
- Cosine similarity: Ratio of similarity between two strings using Levenshtein ratio
- Jaccard similarity: Similarity score using Jaro-Winkler distance

Functions:
    calc_levenstein_distance(word1, word2): Calculate Levenshtein edit distance
    calc_cosine_distance(word1, word2): Calculate cosine similarity using Levenshtein ratio
    calc_jaccard_distance(word1, word2): Calculate Jaccard similarity using Jaro-Winkler
"""

def calc_levenstein_distance(word1, word2):
    """Calculate Levenshtein edit distance between two words.
    Args:
        word1 (str): The first word.
        word2 (str): The second word.
    Returns:
        int: The Levenshtein edit distance between the two words.
    """
    return Levenshtein.distance(word1, word2, weights = (2,2,1))

def calc_cosine_distance(word1, word2):
    """Calculate cosine similarity between two words.
    Args:
        word1 (str): The first word.
        word2 (str): The second word.
    Returns:
        float: The cosine similarity between the two words.
    """
    return Levenshtein.ratio(word1, word2)

def calc_jaccard_distance(word1, word2):
    """Calculate Jaccard similarity between two words.
    Args:
        word1 (str): The first word.
        word2 (str): The second word.
    Returns:
        float: The Jaccard similarity between the two words.
    """
    return Levenshtein.jaro_winkler(word1, word2)