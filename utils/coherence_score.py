import json
import math
import os
from itertools import combinations
from collections import defaultdict
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# --- UMassCoherence Class from umass_test.py ---

class UMassCoherence:
    def __init__(self, documents, epsilon=1e-12):
        """
        Initialize U_mass coherence calculator

        Args:
            documents: List of documents, where each document is a list of words
            epsilon: Small value to avoid log(0)
        """
        self.documents = documents
        self.epsilon = epsilon
        self.word_doc_freq = defaultdict(set)
        self.cooccur_freq = defaultdict(lambda: defaultdict(int))

        # Build word-document frequency and co-occurrence matrices
        self._build_frequencies()

    def _build_frequencies(self):
        """Build word-document frequency and co-occurrence frequency dictionaries"""
        for doc_id, doc in enumerate(self.documents):
            # Get unique words in document
            unique_words = set(doc)

            # Track which documents contain each word
            for word in unique_words:
                self.word_doc_freq[word].add(doc_id)

            # Track co-occurrences
            unique_words_list = list(unique_words)
            for i in range(len(unique_words_list)):
                for j in range(i + 1, len(unique_words_list)):
                    word1, word2 = unique_words_list[i], unique_words_list[j]
                    # Store co-occurrences symmetrically
                    self.cooccur_freq[word1][word2] += 1
                    self.cooccur_freq[word2][word1] += 1

    def calculate_umass_coherence(self, topic_words, top_n=10):
        """
        Calculate U_mass coherence for a topic

        Args:
            topic_words: List of (word, score) tuples for the topic or dict of word scores
            top_n: Number of top words to consider

        Returns:
            U_mass coherence score
        """
        # Get top N words
        if isinstance(topic_words, dict):
            # Sort by score and get top N
            sorted_words = sorted(topic_words.items(), key=lambda x: x[1], reverse=True)
            top_words = []
            for word, score in sorted_words[:top_n]:
                # Handle words with "/" separator (take first part)
                if "/" in word:
                    words = word.split("/")
                    top_words.append(words[0].strip())
                else:
                    top_words.append(word)
        else:
            # Assume it's already a list of words
            top_words = topic_words[:top_n]

        coherence_score = 0.0
        pair_count = 0

        # Calculate pairwise coherence
        for i in range(1, len(top_words)):
            for j in range(i):
                word_i = top_words[i]
                word_j = top_words[j]

                # Get document frequencies
                D_wi = len(self.word_doc_freq.get(word_i, set()))
                D_wj = len(self.word_doc_freq.get(word_j, set()))

                # Get co-occurrence frequency
                D_wi_wj = self.cooccur_freq.get(word_i, {}).get(word_j, 0)

                # Calculate U_mass score for this pair
                if D_wi > 0 and D_wj > 0 and D_wi_wj > 0:
                    # U_mass formula: log((D(wi, wj) + epsilon) / D(wi))
                    score = math.log((D_wi_wj + self.epsilon) / D_wj)
                    coherence_score += score
                    pair_count += 1

        # Return average coherence
        if pair_count > 0:
            return coherence_score / pair_count
        else:
            return 0.0

    def calculate_all_topics_coherence(self, topics_dict, top_n=10):
        """
        Calculate U_mass coherence for all topics

        Args:
            topics_dict: Dictionary of topics with word scores
            top_n: Number of top words to consider per topic

        Returns:
            Dictionary of coherence scores for each topic and average coherence
        """
        topic_coherences = {}
        coherence_values = []

        for topic_name, topic_words in topics_dict.items():
            coherence_score = self.calculate_umass_coherence(topic_words, top_n)
            topic_coherences[f"{topic_name}_coherence"] = coherence_score
            coherence_values.append(coherence_score)

        average_coherence = sum(coherence_values) / len(coherence_values) if coherence_values else 0.0
        
        return {
            "topic_coherences": topic_coherences,
            "average_coherence": average_coherence
        }

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
            topic_coherence = sum(pmi_values) / len(pmi_values)
            topic_coherences[f"{topic_id}_coherence"] = topic_coherence
            total_coherence_sum += topic_coherence
            valid_topics_count += 1

    average_coherence = total_coherence_sum / valid_topics_count if valid_topics_count > 0 else 0.0
    return {
        "topic_coherences": topic_coherences, 
        "average_coherence": average_coherence
    }

def calculate_coherence_scores(topic_word_scores, output_dir=None, table_name=None, column_name=None, cleaned_data=None):
    print("Calculating coherence scores...")
    
    # Calculate U-Mass using the class-based implementation
    coherence_scores = u_mass(topic_word_scores, table_name=table_name, column_name=column_name, documents=cleaned_data)
    
    if coherence_scores is None:
        print("Error: Could not calculate coherence scores.")
        return None
    
    print(f"U-Mass Average Coherence (Class-based): {coherence_scores['average_coherence']:.4f}")
    for topic, score in coherence_scores['topic_coherences'].items():
        print(f"{topic}: {score:.4f}")

    results = {"class_based": coherence_scores}
    
    # Add Gensim comparison if cleaned_data is available
    gensim_cal = True
    if cleaned_data and gensim_cal:
        try:
            # Ensure cleaned_data is list of lists
            if isinstance(cleaned_data[0], str):
                cleaned_data = [doc.split() for doc in cleaned_data]
            
            # Prepare the data required by Gensim
            topics_list, dictionary, corpus = prepare_gensim_data(topic_word_scores, cleaned_data)

            gensim_results = CoherenceModel(
                topics=topics_list,
                texts=cleaned_data,  # Use the tokenized documents directly, not the corpus
                dictionary=dictionary,
                coherence='u_mass'
            )
            umass_gensim = gensim_results.get_coherence()
            umass_per_topic = gensim_results.get_coherence_per_topic()
            results["gensim"] = {
                "umass_average": umass_gensim,
                "umass_per_topic": umass_per_topic.tolist() if hasattr(umass_per_topic, 'tolist') else umass_per_topic
            }
            print(f"Gensim U-Mass: {umass_gensim:.4f}")
            print(f"Difference (Class - Gensim): {coherence_scores['average_coherence'] - umass_gensim:.4f}")
            
        except Exception as e:
            print(f"Warning: Could not calculate Gensim coherence: {str(e)}")
        
    if output_dir and table_name:
        coherence_file = os.path.join(output_dir, f"{table_name}_coherence_scores.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(coherence_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Coherence scores saved to: {coherence_file}")

    return results

def u_mass(topics_json, table_name=None, column_name=None, documents=None, epsilon=1e-12):
    """
    Calculates the U-Mass coherence score for topics using the UMassCoherence class
    
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
    # Prepare documents
    if documents is not None:
        final_documents = documents
    else:   
        if table_name and column_name:
            final_documents = get_documents_from_db(table_name, column_name)
            if final_documents is None:  # If database fetch failed, use topics as documents
                final_documents = []
                for topic_id, word_scores in topics_json.items():
                    doc = list(word_scores.keys())
                    final_documents.append(doc)
        else:
            # Create pseudo-documents from topics
            final_documents = []
            for topic_id, word_scores in topics_json.items():
                doc = list(word_scores.keys())
                final_documents.append(doc)
    
    # Ensure documents are tokenized (list of lists)
    if final_documents and isinstance(final_documents[0], str):
        # Convert strings to lists of words
        final_documents = [doc.split() for doc in final_documents]
    
    if len(topics_json) == 0:
        print("Error: No topics found in the data.")
        return None
    
    if not final_documents:
        print("Error: No documents available for coherence calculation.")
        return None
    
    # Initialize UMassCoherence calculator
    umass_calc = UMassCoherence(final_documents, epsilon=epsilon)
    
    # Calculate coherence scores for all topics
    return umass_calc.calculate_all_topics_coherence(topics_json)


def prepare_gensim_data(topics_json, documents):
    """
    Prepare data for Gensim's CoherenceModel

    Args:
        topics_json (dict): Dictionary containing topics and their word scores
        documents (list): List of tokenized documents (each document should be a list of tokens)

    Returns:
        tuple: (topics_list, dictionary, corpus)
    """
    # Prepare topics list
    topics_list = []
    for topic_id, word_scores in topics_json.items():
        # if word is like this "word1 / word2" get the word1
        top_words = []
        for word, score in word_scores.items(): 
            if "/" in word:
                words = word.split("/")
                top_words.append(words[0].strip())  # Take the first part before the slash
            else:
                top_words.append(word)
        topics_list.append(top_words)

    # Ensure documents are properly tokenized
    if not documents or len(documents) == 0:
        raise ValueError("No documents provided")

    tokenized_documents = documents

    # Create dictionary and corpus (corpus not used for u_mass but may be useful for other coherence measures)
    dictionary = Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

    return topics_list, dictionary, corpus



# --- Example Usage ---
if __name__ == '__main__':
    # Create sample topics and documents for comparison
    print("=== Coherence Score Comparison: Manual vs Gensim ===\n")
    
    # Sample topics (word scores)
    sample_topics = {
        "topic_0": {
            "machine": 0.8,
            "learning": 0.7,
            "algorithm": 0.6,
            "data": 0.5,
            "model": 0.4
        },
        "topic_1": {
            "neural": 0.9,
            "network": 0.8,
            "deep": 0.7,
            "learning": 0.6,
            "artificial": 0.5
        },
        "topic_2": {
            "natural": 0.8,
            "language": 0.7,
            "processing": 0.6,
            "text": 0.5,
            "nlp": 0.4
        }
    }
    
    # Sample documents (tokenized)
    sample_documents = [
        ["machine", "learning", "algorithm", "data", "science", "model", "prediction"],
        ["neural", "network", "deep", "learning", "artificial", "intelligence"],
        ["natural", "language", "processing", "text", "nlp", "analysis"],
        ["machine", "learning", "model", "training", "data", "algorithm"],
        ["deep", "neural", "network", "artificial", "intelligence", "learning"],
        ["text", "processing", "natural", "language", "nlp", "analysis"],
        ["data", "science", "machine", "learning", "model", "algorithm"],
        ["artificial", "intelligence", "neural", "network", "deep"],
        ["language", "processing", "text", "natural", "nlp"],
        ["learning", "machine", "algorithm", "data", "model"]
    ]
    
    print("Sample Topics:")
    for topic_id, words in sample_topics.items():
        top_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"{topic_id}: {[word for word, score in top_words]}")
    print(f"\nNumber of documents: {len(sample_documents)}")
    print(f"Sample document: {sample_documents[0]}\n")
    
    # === Manual Implementations ===
    print("1. Manual Implementation Results:")
    print("-" * 40)
    
    # Calculate U-Mass coherence manually
    print("U-Mass Coherence (Manual):")
    umass_manual = u_mass(sample_topics, documents=sample_documents)
    print(f"Average U-Mass: {umass_manual['average_coherence']:.4f}")
    for topic, score in umass_manual['topic_coherences'].items():
        print(f"  {topic}: {score:.4f}")
    
    print("\nUCI Coherence (Manual):")
    uci_manual = c_uci(sample_topics, documents=sample_documents)
    print(f"Average UCI: {uci_manual['average_coherence']:.4f}")
    for topic, score in uci_manual['topic_coherences'].items():
        print(f"  {topic}: {score:.4f}")
    
    # === Gensim Implementation ===
    print("\n\n2. Gensim Implementation Results:")
    print("-" * 40)
    
    try:
        # Prepare data for Gensim
        
        topics_list, dictionary, corpus = prepare_gensim_data(sample_topics, sample_documents)
        
        print("Gensim U-Mass Coherence:")
        # Calculate U-Mass using Gensim
        cm_umass = CoherenceModel(
            topics=topics_list,
            texts=sample_documents,
            dictionary=dictionary,
            coherence='u_mass'
        )
        umass_gensim = cm_umass.get_coherence()
        umass_per_topic = cm_umass.get_coherence_per_topic()
        
        print(f"Average U-Mass: {umass_gensim:.4f}")
        for i, score in enumerate(umass_per_topic):
            print(f"  topic_{i}_coherence: {score:.4f}")
        
        print("\nGensim C_V Coherence:")
        # Calculate C_V using Gensim (alternative measure)
        cm_cv = CoherenceModel(
            topics=topics_list,
            texts=sample_documents,
            dictionary=dictionary,
            coherence='c_v'
        )
        cv_gensim = cm_cv.get_coherence()
        cv_per_topic = cm_cv.get_coherence_per_topic()
        
        print(f"Average C_V: {cv_gensim:.4f}")
        for i, score in enumerate(cv_per_topic):
            print(f"  topic_{i}_coherence: {score:.4f}")
        
        print("\nGensim C_UCI Coherence:")
        # Calculate C_UCI using Gensim
        cm_cuci = CoherenceModel(
            topics=topics_list,
            texts=sample_documents,
            dictionary=dictionary,
            coherence='c_uci'
        )
        cuci_gensim = cm_cuci.get_coherence()
        cuci_per_topic = cm_cuci.get_coherence_per_topic()
        
        print(f"Average C_UCI: {cuci_gensim:.4f}")
        for i, score in enumerate(cuci_per_topic):
            print(f"  topic_{i}_coherence: {score:.4f}")
        
    except Exception as e:
        print(f"Error with Gensim calculation: {str(e)}")
    
    # === Comparison ===
    print("\n\n3. Comparison Summary:")
    print("-" * 40)
    
    try:
        print(f"U-Mass Difference (Manual - Gensim): {umass_manual['average_coherence'] - umass_gensim:.4f}")
        print(f"Manual U-Mass: {umass_manual['average_coherence']:.4f}")
        print(f"Gensim U-Mass: {umass_gensim:.4f}")
        
        print(f"\nManual UCI: {uci_manual['average_coherence']:.4f}")
        print(f"Gensim C_UCI: {cuci_gensim:.4f}")
        print(f"UCI Difference (Manual - Gensim): {uci_manual['average_coherence'] - cuci_gensim:.4f}")
        
        print("\nNote: Small differences are expected due to different implementation details,")
        print("preprocessing steps, and calculation methods between manual and Gensim implementations.")
        
    except:
        print("Could not complete full comparison due to Gensim calculation errors.")
    
    # === Test with different document sets ===
    print("\n\n4. Testing with Different Document Characteristics:")
    print("-" * 40)
    
    # Test with highly coherent documents
    coherent_docs = [
        ["machine", "learning", "algorithm", "data"],
        ["machine", "learning", "model", "data"],
        ["algorithm", "data", "machine", "learning"],
        ["neural", "network", "deep", "learning"],
        ["neural", "network", "artificial", "intelligence"],
        ["deep", "learning", "neural", "network"]
    ]
    
    print("Testing with highly coherent documents:")
    umass_coherent = u_mass(sample_topics, documents=coherent_docs)
    print(f"U-Mass (coherent docs): {umass_coherent['average_coherence']:.4f}")
    
    # Test with random documents
    random_docs = [
        ["apple", "car", "house", "computer"],
        ["tree", "phone", "book", "music"],
        ["water", "fire", "earth", "air"],
        ["cat", "dog", "bird", "fish"],
        ["red", "blue", "green", "yellow"]
    ]
    
    print("Testing with random/incoherent documents:")
    umass_random = u_mass(sample_topics, documents=random_docs)
    print(f"U-Mass (random docs): {umass_random['average_coherence']:.4f}")
    

    print(f"\nCoherence difference: {umass_coherent['average_coherence'] - umass_random['average_coherence']:.4f}")
    print("Higher coherence scores indicate better topic quality.")