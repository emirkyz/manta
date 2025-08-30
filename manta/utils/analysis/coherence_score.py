import json
import math
import os
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import multiprocessing as mp

def fix_multiprocessing_fork():
      try:
          mp.set_start_method('fork', force=True)
      except RuntimeError:
          pass  # Already set

# --- PerplexityScorer Class ---

class PerplexityScorer:
    def __init__(self, documents, topic_word_matrix=None, doc_topic_matrix=None, epsilon=1e-12):
        """
        Initialize perplexity calculator for topic models
        
        Args:
            documents: List of documents, where each document is a list of words
            topic_word_matrix: Topic-word probability matrix (H matrix from NMF) - shape: (n_topics, n_words)
            doc_topic_matrix: Document-topic probability matrix (W matrix from NMF) - shape: (n_docs, n_topics)
            epsilon: Small value to avoid log(0)
        """
        self.documents = documents
        self.topic_word_matrix = topic_word_matrix
        self.doc_topic_matrix = doc_topic_matrix
        self.epsilon = epsilon
        self.vocabulary = None
        self.word_to_idx = {}
        
        if documents:
            self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build vocabulary from documents"""
        vocabulary = set()
        for doc in self.documents:
            if isinstance(doc, list):
                vocabulary.update(doc)
            else:
                vocabulary.update(doc.split())
        
        self.vocabulary = list(vocabulary)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
    
    def calculate_document_perplexity(self, document, doc_idx=None):
        """
        Calculate perplexity for a single document
        
        Args:
            document: List of words in the document or string
            doc_idx: Index of document in doc_topic_matrix (if available)
            
        Returns:
            Perplexity score for the document
        """
        if isinstance(document, str):
            words = document.split()
        else:
            words = document
            
        if not words:
            return float('inf')
        
        log_likelihood = 0.0
        word_count = 0
        
        # If we have topic matrices, use them for calculation
        if self.topic_word_matrix is not None and self.doc_topic_matrix is not None and doc_idx is not None:
            doc_topic_dist = self.doc_topic_matrix[doc_idx] if doc_idx < len(self.doc_topic_matrix) else None
            
            if doc_topic_dist is not None:
                for word in words:
                    if word in self.word_to_idx:
                        word_idx = self.word_to_idx[word]
                        if word_idx < self.topic_word_matrix.shape[1]:
                            # P(word|document) = sum_t P(word|topic_t) * P(topic_t|document)
                            word_prob = 0.0
                            for topic_idx in range(len(doc_topic_dist)):
                                if topic_idx < self.topic_word_matrix.shape[0]:
                                    topic_word_prob = self.topic_word_matrix[topic_idx, word_idx]
                                    doc_topic_prob = doc_topic_dist[topic_idx]
                                    word_prob += topic_word_prob * doc_topic_prob
                            
                            word_prob = max(word_prob, self.epsilon)
                            log_likelihood += math.log(word_prob)
                            word_count += 1
        else:
            # Fallback: Simple uniform distribution perplexity
            uniform_prob = 1.0 / len(self.vocabulary) if self.vocabulary else self.epsilon
            for word in words:
                if word in self.vocabulary:
                    log_likelihood += math.log(uniform_prob)
                    word_count += 1
        
        if word_count == 0:
            return float('inf')
        
        # Perplexity = exp(-1/N * sum(log P(w_i)))
        avg_log_likelihood = log_likelihood / word_count
        perplexity = math.exp(-avg_log_likelihood)
        
        return perplexity
    
    def calculate_topic_perplexity(self, topic_idx, test_documents=None):
        """
        Calculate perplexity for a specific topic
        
        Args:
            topic_idx: Index of the topic
            test_documents: List of test documents (if None, uses training documents)
            
        Returns:
            Perplexity score for the topic
        """
        if test_documents is None:
            test_documents = self.documents
            
        if self.topic_word_matrix is None or topic_idx >= self.topic_word_matrix.shape[0]:
            return float('inf')
        
        topic_word_dist = self.topic_word_matrix[topic_idx]
        total_log_likelihood = 0.0
        total_words = 0
        
        for doc in test_documents:
            if isinstance(doc, str):
                words = doc.split()
            else:
                words = doc
                
            for word in words:
                if word in self.word_to_idx:
                    word_idx = self.word_to_idx[word]
                    if word_idx < len(topic_word_dist):
                        word_prob = max(topic_word_dist[word_idx], self.epsilon)
                        total_log_likelihood += math.log(word_prob)
                        total_words += 1
        
        if total_words == 0:
            return float('inf')
        
        avg_log_likelihood = total_log_likelihood / total_words
        perplexity = math.exp(-avg_log_likelihood)
        
        return perplexity
    
    def calculate_all_topics_perplexity(self, test_documents=None):
        """
        Calculate perplexity for all topics
        
        Args:
            test_documents: List of test documents (if None, uses training documents)
            
        Returns:
            Dictionary containing perplexity scores for each topic and average perplexity
        """
        if test_documents is None:
            test_documents = self.documents
            
        topic_perplexities = {}
        perplexity_values = []
        
        if self.topic_word_matrix is not None:
            n_topics = self.topic_word_matrix.shape[0]
            
            for topic_idx in range(n_topics):
                topic_perplexity = self.calculate_topic_perplexity(topic_idx, test_documents)
                topic_name = f"topic_{topic_idx}"
                
                # Handle inf and nan values
                if math.isnan(topic_perplexity):
                    topic_perplexity = float('inf')
                
                topic_perplexities[f"{topic_name}_perplexity"] = topic_perplexity
                
                if not math.isinf(topic_perplexity) and not math.isnan(topic_perplexity):
                    perplexity_values.append(topic_perplexity)
        else:
            # Fallback: calculate using simple word frequency
            all_words = set()
            for doc in test_documents:
                if isinstance(doc, str):
                    all_words.update(doc.split())
                else:
                    all_words.update(doc)
            
            if all_words:
                uniform_perplexity = len(all_words)  # Simple baseline
                for topic_idx in range(len(topic_word_scores) if 'topic_word_scores' in globals() else 3):
                    topic_name = f"topic_{topic_idx}"
                    topic_perplexities[f"{topic_name}_perplexity"] = uniform_perplexity
                    perplexity_values.append(uniform_perplexity)
        
        # Calculate average, handling edge cases
        if perplexity_values:
            average_perplexity = sum(perplexity_values) / len(perplexity_values)
        else:
            # If no valid perplexity values, use vocabulary size as baseline
            vocab_size = len(self.vocabulary) if self.vocabulary else len(set(
                word for doc in test_documents for word in (doc.split() if isinstance(doc, str) else doc)
            ))
            average_perplexity = max(vocab_size, 10.0)  # At least 10.0 as minimum baseline
        
        return {
            "topic_perplexities": topic_perplexities,
            "average_perplexity": average_perplexity
        }
    
    def calculate_corpus_perplexity(self, test_documents=None):
        """
        Calculate overall corpus perplexity using document-topic and topic-word distributions
        
        Args:
            test_documents: List of test documents (if None, uses training documents)
            
        Returns:
            Overall corpus perplexity score
        """
        if test_documents is None:
            test_documents = self.documents
            
        if self.topic_word_matrix is None or self.doc_topic_matrix is None:
            # Fallback to simple perplexity calculation
            total_log_likelihood = 0.0
            total_words = 0
            
            for doc in test_documents:
                doc_perplexity = self.calculate_document_perplexity(doc)
                if not math.isinf(doc_perplexity):
                    if isinstance(doc, str):
                        word_count = len(doc.split())
                    else:
                        word_count = len(doc)
                    total_log_likelihood += -math.log(doc_perplexity) * word_count
                    total_words += word_count
            
            if total_words == 0:
                return float('inf')
            
            avg_log_likelihood = total_log_likelihood / total_words
            return math.exp(-avg_log_likelihood)
        
        total_log_likelihood = 0.0
        total_words = 0
        
        # Calculate perplexity using the full model
        for doc_idx, document in enumerate(test_documents):
            if doc_idx < len(self.doc_topic_matrix):
                doc_perplexity = self.calculate_document_perplexity(document, doc_idx)
                if not math.isinf(doc_perplexity):
                    if isinstance(document, str):
                        word_count = len(document.split())
                    else:
                        word_count = len(document)
                    total_log_likelihood += -math.log(doc_perplexity) * word_count
                    total_words += word_count
        
        if total_words == 0:
            return float('inf')
        
        avg_log_likelihood = total_log_likelihood / total_words
        return math.exp(-avg_log_likelihood)


# --- TopicDiversityScorer Class ---

class TopicDiversityScorer:
    def __init__(self, topic_word_scores, top_words=10):
        """
        Initialize topic diversity calculator
        
        Args:
            topic_word_scores (dict): Dictionary containing topics and their word scores
            top_words (int): Number of top words to consider per topic for diversity calculation
        """
        self.topic_word_scores = topic_word_scores
        self.top_words = top_words
        self.topic_word_lists = {}
        self.all_words = set()
        
        if topic_word_scores:
            self._prepare_topic_words()
    
    def _prepare_topic_words(self):
        """Prepare top words for each topic and build vocabulary"""
        for topic_id, word_scores in self.topic_word_scores.items():
            # Sort words by score and get top N
            if isinstance(word_scores, dict):
                sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
                top_words = []
                for word, score in sorted_words[:self.top_words]:
                    # Handle words with "/" separator (take first part)
                    if "/" in word:
                        words = word.split("/")
                        top_words.append(words[0].strip())
                    else:
                        top_words.append(word)
                self.topic_word_lists[topic_id] = top_words
            else:
                # Assume it's already a list of words
                self.topic_word_lists[topic_id] = list(word_scores)[:self.top_words]
            
            # Add to overall vocabulary
            self.all_words.update(self.topic_word_lists[topic_id])
    
    def calculate_proportion_unique_words(self):
        """
        Calculate Proportion of Unique Words (PUW) - Intra-topic diversity
        
        Returns:
            float: Ratio of unique words across all topics vs total words
        """
        if not self.topic_word_lists:
            return 0.0
        
        # Count total words across all topics
        total_word_instances = sum(len(words) for words in self.topic_word_lists.values())
        
        # Count unique words
        unique_words = len(self.all_words)
        
        if total_word_instances == 0:
            return 0.0
        
        return unique_words / total_word_instances
    
    def calculate_jaccard_similarity(self, topic1_words, topic2_words):
        """
        Calculate Jaccard similarity between two topic word lists
        
        Args:
            topic1_words (list): Words from first topic
            topic2_words (list): Words from second topic
            
        Returns:
            float: Jaccard similarity (0-1)
        """
        set1 = set(topic1_words)
        set2 = set(topic2_words)
        
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_cosine_similarity(self, topic1_words, topic2_words):
        """
        Calculate cosine similarity between two topic word lists using binary vectors
        
        Args:
            topic1_words (list): Words from first topic
            topic2_words (list): Words from second topic
            
        Returns:
            float: Cosine similarity (0-1)
        """
        set1 = set(topic1_words)
        set2 = set(topic2_words)
        
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        
        # Create binary vectors based on vocabulary
        vocab = list(self.all_words)
        if len(vocab) == 0:
            return 0.0
        
        vector1 = np.array([1.0 if word in set1 else 0.0 for word in vocab])
        vector2 = np.array([1.0 if word in set2 else 0.0 for word in vocab])
        
        # Calculate cosine similarity
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_pairwise_similarities(self, similarity_func):
        """
        Calculate pairwise similarities between all topic pairs
        
        Args:
            similarity_func: Function to calculate similarity between two topic word lists
            
        Returns:
            dict: Dictionary of pairwise similarities
        """
        similarities = {}
        topic_ids = list(self.topic_word_lists.keys())
        
        for i in range(len(topic_ids)):
            for j in range(i + 1, len(topic_ids)):
                topic1_id = topic_ids[i]
                topic2_id = topic_ids[j]
                
                similarity = similarity_func(
                    self.topic_word_lists[topic1_id],
                    self.topic_word_lists[topic2_id]
                )
                
                pair_name = f"{topic1_id}_vs_{topic2_id}"
                similarities[pair_name] = similarity
        
        return similarities
    
    def calculate_average_pairwise_diversity(self, similarity_func):
        """
        Calculate average pairwise diversity (1 - similarity)
        
        Args:
            similarity_func: Function to calculate similarity between two topic word lists
            
        Returns:
            float: Average diversity score (higher = more diverse)
        """
        similarities = self.calculate_pairwise_similarities(similarity_func)
        
        if not similarities:
            return 0.0
        
        # Convert similarities to diversities (1 - similarity)
        diversities = [1.0 - sim for sim in similarities.values()]
        
        return sum(diversities) / len(diversities)
    
    def find_most_similar_topics(self, similarity_func, top_k=3):
        """
        Find the most similar topic pairs
        
        Args:
            similarity_func: Function to calculate similarity
            top_k (int): Number of top similar pairs to return
            
        Returns:
            list: List of (pair_name, similarity_score) tuples
        """
        similarities = self.calculate_pairwise_similarities(similarity_func)
        
        # Sort by similarity (descending)
        sorted_pairs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_pairs[:top_k]
    
    def find_most_diverse_topics(self, similarity_func, top_k=3):
        """
        Find the most diverse (least similar) topic pairs
        
        Args:
            similarity_func: Function to calculate similarity
            top_k (int): Number of top diverse pairs to return
            
        Returns:
            list: List of (pair_name, diversity_score) tuples
        """
        similarities = self.calculate_pairwise_similarities(similarity_func)
        
        # Convert to diversity and sort by diversity (descending)
        diversities = [(pair, 1.0 - sim) for pair, sim in similarities.items()]
        sorted_pairs = sorted(diversities, key=lambda x: x[1], reverse=True)
        
        return sorted_pairs[:top_k]
    
    def calculate_all_diversity_metrics(self):
        """
        Calculate all diversity metrics
        
        Returns:
            dict: Comprehensive diversity analysis
        """
        if not self.topic_word_lists:
            return {
                "proportion_unique_words": 0.0,
                "average_jaccard_diversity": 0.0,
                "average_cosine_diversity": 0.0,
                "jaccard_similarities": {},
                "cosine_similarities": {},
                "diversity_summary": {
                    "most_similar_topics": [],
                    "most_diverse_topics": [],
                    "overall_diversity_score": 0.0
                }
            }
        
        # Calculate intra-topic diversity
        puw = self.calculate_proportion_unique_words()
        
        # Calculate pairwise diversities
        jaccard_diversity = self.calculate_average_pairwise_diversity(self.calculate_jaccard_similarity)
        cosine_diversity = self.calculate_average_pairwise_diversity(self.calculate_cosine_similarity)
        
        # Get pairwise similarities for detailed analysis
        jaccard_similarities = self.calculate_pairwise_similarities(self.calculate_jaccard_similarity)
        cosine_similarities = self.calculate_pairwise_similarities(self.calculate_cosine_similarity)
        
        # Find most similar and diverse topic pairs
        most_similar = self.find_most_similar_topics(self.calculate_jaccard_similarity, top_k=3)
        most_diverse = self.find_most_diverse_topics(self.calculate_jaccard_similarity, top_k=3)
        
        # Overall diversity score (average of different measures)
        overall_diversity = (puw + jaccard_diversity + cosine_diversity) / 3.0
        
        return {
            "proportion_unique_words": puw,
            "average_jaccard_diversity": jaccard_diversity,
            "average_cosine_diversity": cosine_diversity,
            "jaccard_similarities": jaccard_similarities,
            "cosine_similarities": cosine_similarities,
            "diversity_summary": {
                "most_similar_topics": [pair[0] for pair in most_similar],
                "most_diverse_topics": [pair[0] for pair in most_diverse],
                "overall_diversity_score": overall_diversity
            }
        }


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
    base_dir = Path(__file__).parent.resolve()
    instance_path = base_dir / ".." / "instance"
    db_path = instance_path / "scopus.db"
    
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

def calculate_coherence_scores(topic_word_scores, output_dir=None, table_name=None, column_name=None, cleaned_data=None, topic_word_matrix=None, doc_topic_matrix=None):
    print("Calculating coherence scores...")
    fix_multiprocessing_fork()

    u_mass_manual = False
    if u_mass_manual:
        # Calculate U-Mass using the class-based implementation
        coherence_scores = u_mass(topic_word_scores, table_name=table_name, column_name=column_name, documents=cleaned_data)

        if coherence_scores is None:
            print("Error: Could not calculate coherence scores.")
            return None

        print(f"U-Mass Average Coherence (Class-based): {coherence_scores['average_coherence']:.4f}")
        for topic, score in coherence_scores['topic_coherences'].items():
            print(f"{topic}: {score:.4f}")

        results = {"class_based": coherence_scores}
    else:
        results = {}
    
    # Calculate Perplexity scores
    print("Calculating perplexity scores...")
    perplexity_scores = calculate_perplexity_scores(
        topic_word_scores, 
        documents=cleaned_data,
        topic_word_matrix=topic_word_matrix,
        doc_topic_matrix=doc_topic_matrix
    )
    
    if perplexity_scores is not None:
        results["perplexity"] = perplexity_scores
        
        # Display average perplexity
        avg_perp = perplexity_scores['average_perplexity']
        if math.isinf(avg_perp) or math.isnan(avg_perp):
            print(f"Average Perplexity: N/A (no valid calculations)")
        else:
            print(f"Average Perplexity: {avg_perp:.4f}")
        
        # Display corpus perplexity
        corpus_perp = perplexity_scores['corpus_perplexity']
        if math.isinf(corpus_perp) or math.isnan(corpus_perp):
            print(f"Corpus Perplexity: N/A (no valid calculations)")
        else:
            print(f"Corpus Perplexity: {corpus_perp:.4f}")
        
        # Display topic perplexities
        valid_topic_count = 0
        for topic, score in perplexity_scores['topic_perplexities'].items():
            if not math.isinf(score) and not math.isnan(score):
                print(f"{topic}: {score:.4f}")
                valid_topic_count += 1
        
        if valid_topic_count == 0:
            print("No valid topic perplexity scores calculated.")
    else:
        print("Warning: Could not calculate perplexity scores.")
    
    # Calculate Diversity scores
    print("Calculating topic diversity scores...")
    diversity_scores = calculate_diversity_scores(topic_word_scores, top_words=10)
    
    if diversity_scores is not None:
        results["diversity"] = diversity_scores
        print(f"Overall Topic Diversity Score: {diversity_scores['diversity_summary']['overall_diversity_score']:.4f}")
    else:
        print("Warning: Could not calculate diversity scores.")
    
    # Add Gensim comparison if cleaned_data is available
    gensim_cal = True
    coherence_method = "c_v"
    if cleaned_data and gensim_cal:
        try:
            # Check if cleaned_data is already tokenized (list of lists) or needs tokenization (list of strings)
            if cleaned_data and isinstance(cleaned_data[0], list):
                # Already tokenized
                cleaned_data_token = cleaned_data
            elif cleaned_data and isinstance(cleaned_data[0], str):
                # Need to tokenize
                cleaned_data_token = [doc.split() for doc in cleaned_data]
            else:
                raise ValueError("cleaned_data must be a list of strings or a list of lists")
                
            # Prepare the data required by Gensim
            topics_list, dictionary, corpus = prepare_gensim_data(topic_word_scores, cleaned_data_token)

            gensim_results = CoherenceModel(
                topics=topics_list,
                texts=cleaned_data_token,  # Use the tokenized documents directly, not the corpus
                dictionary=dictionary,
                coherence=coherence_method,
            )
            umass_gensim = gensim_results.get_coherence()
            umass_per_topic = gensim_results.get_coherence_per_topic()
            # Create dictionary with topic-specific coherence scores
            topic_coherence_dict = {}
            for i, score in enumerate(umass_per_topic):
                topic_coherence_dict[f"konu {i+1}"] = score.tolist() if hasattr(score, 'tolist') else score

            results["gensim"] = {
                f"{coherence_method}_average": umass_gensim,
                f"{coherence_method}_per_topic": topic_coherence_dict
            }
            print(f"Gensim {coherence_method} Average: {umass_gensim:.4f}")
            #print(f"Difference (Class - Gensim): {coherence_scores['average_coherence'] - umass_gensim:.4f}")
            
        except Exception as e:
            print(f"Warning: Could not calculate Gensim coherence: {str(e)}")
        
    if output_dir and table_name:
        output_path = Path(output_dir)
        coherence_file = output_path / f"{table_name}_coherence_scores.json"
        output_path.mkdir(parents=True, exist_ok=True)
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


def calculate_perplexity_scores(topic_word_scores, documents=None, topic_word_matrix=None, doc_topic_matrix=None, epsilon=1e-12):
    """
    Calculate perplexity scores for topics using the PerplexityScorer class
    
    Args:
        topic_word_scores (dict): Dictionary containing topics and their word scores  
        documents (list, optional): List of documents for perplexity calculation
        topic_word_matrix (numpy.ndarray, optional): Topic-word probability matrix (H matrix from NMF)
        doc_topic_matrix (numpy.ndarray, optional): Document-topic probability matrix (W matrix from NMF)  
        epsilon (float): Small value to prevent log(0)
        
    Returns:
        dict: Dictionary containing topic perplexities and average perplexity
    """
    if len(topic_word_scores) == 0:
        print("Error: No topics found in the data.")
        return None
    
    # Prepare documents
    if documents is None:
        # Create pseudo-documents from topics if no documents provided
        documents = []
        for topic_id, word_scores in topic_word_scores.items():
            doc = list(word_scores.keys())
            documents.append(doc)
    
    # Ensure documents are tokenized (list of lists)
    if documents and isinstance(documents[0], str):
        # Convert strings to lists of words
        documents = [doc.split() for doc in documents]
    
    if not documents:
        print("Error: No documents available for perplexity calculation.")
        return None
    
    # If no matrices provided, create simple fallback perplexity based on word frequencies
    if topic_word_matrix is None:
        # Build vocabulary from documents
        all_doc_words = set()
        for doc in documents:
            all_doc_words.update(doc)
        
        # Build vocabulary from topics  
        all_topic_words = set()
        for topic_words in topic_word_scores.values():
            all_topic_words.update(topic_words.keys())
        
        # Calculate vocabulary overlap
        vocab_overlap = all_doc_words.intersection(all_topic_words)
        total_vocab = all_doc_words.union(all_topic_words)
        
        if len(vocab_overlap) == 0:
            print("Warning: No vocabulary overlap between topics and documents.")
            # Use total vocabulary size as baseline perplexity
            baseline_perplexity = len(total_vocab) if total_vocab else 100.0
        else:
            # Use inverse of overlap ratio as baseline
            overlap_ratio = len(vocab_overlap) / len(total_vocab)
            baseline_perplexity = 1.0 / max(overlap_ratio, epsilon)
        
        # Create simple topic perplexities
        topic_perplexities = {}
        for topic_id in topic_word_scores.keys():
            topic_perplexities[f"{topic_id}_perplexity"] = baseline_perplexity
        
        return {
            "topic_perplexities": topic_perplexities,
            "average_perplexity": baseline_perplexity,
            "corpus_perplexity": baseline_perplexity
        }
    
    # Initialize PerplexityScorer with matrices
    perplexity_calc = PerplexityScorer(
        documents, 
        topic_word_matrix=topic_word_matrix, 
        doc_topic_matrix=doc_topic_matrix, 
        epsilon=epsilon
    )
    
    # Calculate perplexity scores for all topics
    topic_results = perplexity_calc.calculate_all_topics_perplexity(documents)
    
    # Also calculate corpus-level perplexity
    corpus_perplexity = perplexity_calc.calculate_corpus_perplexity(documents)
    
    # Handle nan/inf values in corpus perplexity
    if math.isnan(corpus_perplexity) or math.isinf(corpus_perplexity):
        # Use average of topic perplexities as fallback
        valid_topic_perps = [p for p in topic_results["topic_perplexities"].values() 
                           if not math.isnan(p) and not math.isinf(p)]
        if valid_topic_perps:
            corpus_perplexity = sum(valid_topic_perps) / len(valid_topic_perps)
        else:
            corpus_perplexity = len(perplexity_calc.vocabulary) if perplexity_calc.vocabulary else 50.0
    
    # Combine results
    results = {
        "topic_perplexities": topic_results["topic_perplexities"],
        "average_perplexity": topic_results["average_perplexity"],
        "corpus_perplexity": corpus_perplexity
    }
    
    return results


def calculate_diversity_scores(topic_word_scores, top_words=10):
    """
    Calculate topic diversity scores using the TopicDiversityScorer class
    
    Args:
        topic_word_scores (dict): Dictionary containing topics and their word scores
        top_words (int): Number of top words to consider per topic
        
    Returns:
        dict: Dictionary containing all diversity metrics
    """
    if len(topic_word_scores) == 0:
        print("Error: No topics found for diversity calculation.")
        return None
    
    # Initialize TopicDiversityScorer
    diversity_calc = TopicDiversityScorer(topic_word_scores, top_words=top_words)
    
    # Calculate all diversity metrics
    diversity_results = diversity_calc.calculate_all_diversity_metrics()
    
    return diversity_results


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



def main():
    """Main function for example usage and testing"""
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
        ["machine", "learning", "model", " spotraining", "data", "algorithm"],
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
            coherence='u_mass',
            processes=1  # Disable multiprocessing to prevent restart issues
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
            coherence='c_v',
            processes=1  # Disable multiprocessing to prevent restart issues
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
            coherence='c_uci',
            processes=1  # Disable multiprocessing to prevent restart issues
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
    
    # === Test Perplexity Scores ===
    print("\n\n5. Testing Perplexity Scores:")
    print("-" * 40)
    
    print("Perplexity with original documents:")
    perplexity_original = calculate_perplexity_scores(sample_topics, documents=sample_documents)
    print(f"Average Perplexity: {perplexity_original['average_perplexity']:.4f}")
    print(f"Corpus Perplexity: {perplexity_original['corpus_perplexity']:.4f}")
    for topic, score in perplexity_original['topic_perplexities'].items():
        if not math.isinf(score):
            print(f"  {topic}: {score:.4f}")
    
    print("\nPerplexity with coherent documents:")
    perplexity_coherent = calculate_perplexity_scores(sample_topics, documents=coherent_docs)
    print(f"Average Perplexity: {perplexity_coherent['average_perplexity']:.4f}")
    print(f"Corpus Perplexity: {perplexity_coherent['corpus_perplexity']:.4f}")
    
    print("\nPerplexity with random documents:")
    perplexity_random = calculate_perplexity_scores(sample_topics, documents=random_docs)
    print(f"Average Perplexity: {perplexity_random['average_perplexity']:.4f}")
    print(f"Corpus Perplexity: {perplexity_random['corpus_perplexity']:.4f}")
    
    # Calculate perplexity difference safely
    coherent_avg = perplexity_coherent['average_perplexity']
    random_avg = perplexity_random['average_perplexity']
    
    if (not math.isinf(coherent_avg) and not math.isnan(coherent_avg) and 
        not math.isinf(random_avg) and not math.isnan(random_avg)):
        diff = coherent_avg - random_avg
        print(f"\nPerplexity difference (coherent - random): {diff:.4f}")
    else:
        print(f"\nPerplexity difference: N/A (invalid values)")
    
    print("Lower perplexity scores indicate better topic model fit to the data.")