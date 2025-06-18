import json
import math
import os
from itertools import combinations
from collections import defaultdict
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# --- Helper Functions for Adapted Co-occurrence ---



def p_word_pair(word1,word2,documents):
    """
    Calculates the probability of a word pair in a document
    P(w1,w2) = D(w1,w2) / N
    D(w1,w2) = number of documents containing both word1 and word2
    """
    D_w1_w2 = sum(1 for doc in documents if word1 in doc and word2 in doc)
    N = len(documents)
    return D_w1_w2 / N


def p_word(word,documents):
    """
    Calculates the probability of a word in a document
    P(w) = D(w) / N
    D(w) = number of documents containing word w
    N = total number of documents
    """
    D_w = sum(1 for doc in documents if word in doc)
    N = len(documents)
    return D_w / N

def pmi(word1,word2,documents,epsilon=1e-9):
    """
    Calculates the probability of a word pair in a document
    P(w1,w2) = D(w1,w2) / N
    D(w1,w2) = number of documents containing both word1 and word2
    PMI(w1,w2) = log(P(w1,w2) / (P(w1) * P(w2)))
    """

    p1 = p_word(word1,documents)
    p2 = p_word(word2,documents)
    if p1 == 0 or p2 == 0:
        return "zero_division_error"
    return math.log((p_word_pair(word1,word2,documents) + epsilon) / (p1 * p2))

def get_documents_from_db(table_name, column_name):
    """
    Get documents from SQLite database.
    
    Args:
        table_name (str): Name of the table containing the documents
        column_name (str): Name of the column containing the text
        
    Returns:
        list: List of document texts
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    instance_path = os.path.join(base_dir, "..", "instance")
    db_path = os.path.join(instance_path, "scopus.db")
    
    # Create database engine
    engine = create_engine(f'sqlite:///{db_path}')
    
    try:
        # Read the documents from the database
        query = f"SELECT {column_name} FROM {table_name}"
        df = pd.read_sql_query(query, engine)
        documents = df[column_name].tolist()
        return documents
    except Exception as e:
        print(f"Error reading from database: {str(e)}")
        return None

def c_uci(topics_json, table_name=None, column_name=None, documents=None, epsilon=1e-9):
    """
    Calculates the UCI coherence score for topics
    UCI(w1,w2) = 2 / (N * (N-1)) * sum_i sum_j PMI(w_i,w_j)
    
    Args:
        topics_json (dict): Dictionary containing topics and their word scores
        table_name (str, optional): Name of the table containing the documents
        column_name (str, optional): Name of the column containing the text
        documents (list, optional): List of documents for co-occurrence calculation.
                                  If None and table_name is provided, will fetch from database.
                                  If both None, will use topics themselves as documents.
        epsilon (float): Small value to prevent log(0)
        
    Returns:
        dict: Dictionary containing topic coherences and average coherence
    """
    # If table_name is provided, get documents from database
    if table_name and column_name and documents is None:
        documents = get_documents_from_db(table_name, column_name)
        if documents is None:  # If database fetch failed, use topics as documents
            documents = []
            for topic_id, word_scores in topics_json.items():
                doc = list(word_scores.keys())
                documents.append(doc)
    # If no documents provided and no database access, create pseudo-documents from topics
    elif documents is None:
        documents = []
        for topic_id, word_scores in topics_json.items():
            doc = list(word_scores.keys())
            documents.append(doc)
    
    total_topic_count = len(topics_json)
    if total_topic_count == 0:
        print("Error: No topics found in the data.")
        return None

    topic_coherences = {}
    total_coherence_sum = 0
    valid_topics_count = 0
    
    for topic_id, word_scores in topics_json.items():
        # Sort words by their scores in descending order and take top words
        sorted_words = sorted(word_scores.items(), key=lambda x: float(x[1]), reverse=True)
        top_words = [word for word, _ in sorted_words]
        
        N = len(top_words)
        if N < 2:  # Need at least 2 words to calculate coherence
            continue
            
        word_combinations = combinations(top_words, 2)
        pmi_values = []
        
        for word1, word2 in word_combinations:
            pmi_val = pmi(word1, word2, documents, epsilon)
            if pmi_val != "zero_division_error":
                pmi_values.append(pmi_val)
        
        if pmi_values:  # Only calculate if we have valid PMI values
            # Calculate UCI coherence for this topic
            topic_coherence = 2 * sum(pmi_values) / (N * (N - 1))
            topic_coherences[f"{topic_id}_coherence"] = topic_coherence
            total_coherence_sum += topic_coherence
            valid_topics_count += 1

    average_coherence = total_coherence_sum / valid_topics_count if valid_topics_count > 0 else 0.0
    return {
        "topic_coherences": topic_coherences, 
        "average_coherence": average_coherence
    }

def calculate_coherence_scores(topic_word_scores, output_dir=None, table_name=None, column_name=None):
    """
    Calculate coherence scores (UCI) and optionally save them.

    Args:
        topic_word_scores (dict): Dictionary of topics and their word scores
        output_dir (str, optional): Directory to save the scores. If None, scores won't be saved
        table_name (str, optional): Name of the table/analysis for file naming and database access
        column_name (str, optional): Name of the column containing the document text

    Returns:
        dict: Dictionary containing coherence scores
    """
    print("Calculating coherence scores...")
    
    # Calculate UCI coherence using database documents if table_name is provided
    coherence_scores = c_uci(topic_word_scores, table_name=table_name, column_name=column_name)
    
    # Print coherence scores
    print("\nCoherence Scores:")
    print(f"UCI Average Coherence: {coherence_scores['average_coherence']:.4f}")
    print("\nPer-topic Coherence Scores:")
    for topic, score in coherence_scores['topic_coherences'].items():
        print(f"{topic}: {score:.4f}")

    # Save coherence scores if output directory and table name are provided
    if output_dir and table_name:
        coherence_file = os.path.join(output_dir, f"{table_name}_coherence_scores.json")
        os.makedirs(os.path.dirname(coherence_file), exist_ok=True)
        with open(coherence_file, "w") as f:
            json.dump(coherence_scores, f, indent=4)
        print(f"\nCoherence scores saved to: {coherence_file}")

    return coherence_scores

def doc_cooccurrence(word1, word2, documents):
    """
    Counts number of documents where word1 and word2 co-occur
    """
    return sum(1 for doc in documents if word1 in doc and word2 in doc)

def word_doc_count(word, documents):
    """
    Counts number of documents containing the word
    """
    return sum(1 for doc in documents if word in doc)

def u_mass(topics_json, table_name=None, column_name=None, documents=None, epsilon=1e-9):
    """
    Calculates the U-Mass coherence score for topics
    U-Mass uses log conditional probability: log(P(w2|w1)) = log(P(w1,w2)/P(w1))
    
    Args:
        topics_json (dict): Dictionary containing topics and their word scores
        table_name (str, optional): Name of the table containing the documents
        column_name (str, optional): Name of the column containing the text
        documents (list, optional): List of documents for co-occurrence calculation.
                                  If None and table_name is provided, will fetch from database.
                                  If both None, will use topics themselves as documents.
        epsilon (float): Small value to prevent log(0)
        
    Returns:
        dict: Dictionary containing topic coherences and average coherence
    """
    # If table_name is provided, get documents from database
    if table_name and column_name and documents is None:
        documents = get_documents_from_db(table_name, column_name)
        if documents is None:  # If database fetch failed, use topics as documents
            documents = []
            for topic_id, word_scores in topics_json.items():
                doc = list(word_scores.keys())
                documents.append(doc)
    # If no documents provided and no database access, create pseudo-documents from topics
    elif documents is None:
        documents = []
        for topic_id, word_scores in topics_json.items():
            doc = list(word_scores.keys())
            documents.append(doc)
    
    total_topic_count = len(topics_json)
    if total_topic_count == 0:
        print("Error: No topics found in the data.")
        return None

    topic_coherences = {}
    total_coherence_sum = 0
    valid_topics_count = 0
    
    for topic_id, word_scores in topics_json.items():
        # Sort words by their scores in descending order and take top words
        sorted_words = sorted(word_scores.items(), key=lambda x: float(x[1]), reverse=True)
        top_words = [word for word, _ in sorted_words]
        
        N = len(top_words)
        if N < 2:  # Need at least 2 words to calculate coherence
            continue
            
        # Calculate U-Mass coherence
        coherence_vals = []
        
        # For each word (except the first), calculate coherence with all previous words
        for i in range(1, N):
            for j in range(0, i):
                word2 = top_words[i]
                word1 = top_words[j]
                
                # Get co-occurrence count and individual word count
                co_doc_count = doc_cooccurrence(word1, word2, documents)
                word1_count = word_doc_count(word1, documents)
                
                # Calculate log conditional probability
                if word1_count > 0:
                    score = math.log((co_doc_count + epsilon) / word1_count)
                    coherence_vals.append(score)
        
        if coherence_vals:  # Only calculate if we have valid coherence values
            topic_coherence = sum(coherence_vals) / len(coherence_vals)
            topic_coherences[f"{topic_id}_coherence"] = topic_coherence
            total_coherence_sum += topic_coherence
            valid_topics_count += 1

    average_coherence = total_coherence_sum / valid_topics_count if valid_topics_count > 0 else 0.0
    return {
        "topic_coherences": topic_coherences, 
        "average_coherence": average_coherence
    }

def calculate_coherence_scores(topic_word_scores, output_dir=None, table_name=None, column_name=None):
    """
    Calculate coherence scores (U-Mass) and optionally save them.

    Args:
        topic_word_scores (dict): Dictionary of topics and their word scores
        output_dir (str, optional): Directory to save the scores. If None, scores won't be saved
        table_name (str, optional): Name of the table/analysis for file naming and database access
        column_name (str, optional): Name of the column containing the document text

    Returns:
        dict: Dictionary containing coherence scores
    """
    print("Calculating coherence scores...")
    
    # Calculate U-Mass coherence using database documents if table_name is provided
    coherence_scores = u_mass(topic_word_scores, table_name=table_name, column_name=column_name)
    
    # Print coherence scores
    print("\nCoherence Scores:")
    print(f"U-Mass Average Coherence: {coherence_scores['average_coherence']:.4f}")
    print("\nPer-topic Coherence Scores:")
    for topic, score in coherence_scores['topic_coherences'].items():
        print(f"{topic}: {score:.4f}")

    # Save coherence scores if output directory and table name are provided
    if output_dir and table_name:
        output_folder = os.path.join(output_dir, table_name)
        coherence_file = os.path.join(output_folder, f"{table_name}_coherence_scores.json")
        os.makedirs(os.path.dirname(coherence_file), exist_ok=True)
        with open(coherence_file, "w") as f:
            json.dump(coherence_scores, f, indent=4)
        print(f"\nCoherence scores saved to: {coherence_file}")

    return coherence_scores

# --- Example Usage ---
if __name__ == '__main__':
    # Load the example wordcloud scores
    with open('../Output/FINDINGS_pnmf/FINDINGS_pnmf_wordcloud_scores.json', 'r') as f:
        wordcloud_scores = json.load(f)
    
    # Calculate coherence scores using documents from database
    coherence_results = calculate_coherence_scores(
        wordcloud_scores,
        output_dir="../Output/FINDINGS_pnmf",
        table_name="FINDINGS_pnmf"
    )
    print("\n--- U-Mass Coherence Scores ---")
    print(f"Average Coherence: {coherence_results['average_coherence']:.4f}")
    print("\nPer-topic Coherence Scores:")
    for topic, score in coherence_results['topic_coherences'].items():
        print(f"{topic}: {score:.4f}")
