import json
import math
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import multiprocessing as mp

from manta.utils.analysis.relevance_scorer import get_topic_top_terms


def fix_multiprocessing_fork():
      try:
          mp.set_start_method('fork', force=True)
      except RuntimeError:
          pass  # Already set

# --- PerplexityScorer Class ---



# --- TopicDiversityScorer Class ---

class TopicDiversityScorer:
    def __init__(self, topic_word_scores, top_words=10):
        """
        Initialize topic diversity calculator
        
        Args:
            topic_word_scores (dict): Dictionary containing topics and their word scores
            top_words (int): Number of top words to consider per topic for diversity calculation
        """
        if not isinstance(topic_word_scores, dict):
            raise ValueError("topic_word_scores must be a dictionary")
        
        if top_words <= 0:
            raise ValueError("top_words must be a positive integer")
            
        self.topic_word_scores = topic_word_scores
        self.top_words = top_words
        self.topic_word_lists = {}
        self.all_words = set()
        
        if topic_word_scores:
            self._prepare_topic_words()
        else:
            print("Warning: No topic word scores provided for diversity calculation")
    
    def _prepare_topic_words(self):
        """Prepare top words for each topic and build vocabulary"""
        for topic_id, word_scores in self.topic_word_scores.items():
            if not word_scores:
                print(f"Warning: Topic {topic_id} has no words, skipping")
                continue
                
            # Sort words by score and get top N
            if isinstance(word_scores, dict):
                if not word_scores:
                    print(f"Warning: Topic {topic_id} has empty word scores dictionary")
                    continue
                    
                sorted_words = sorted(word_scores.items(), key=lambda x: float(x[1]) if isinstance(x[1], (int, float, str)) else 0.0, reverse=True)
                top_words = []
                seen_words = set()  # Track duplicates
                
                for word, score in sorted_words:
                    if len(top_words) >= self.top_words:
                        break
                        
                    # Handle words with "/" separator (take first part)
                    if isinstance(word, str) and "/" in word:
                        processed_word = word.split("/")[0].strip().lower()
                    else:
                        processed_word = str(word).strip().lower() if word else ""
                    
                    # Skip empty words and duplicates
                    if processed_word and processed_word not in seen_words:
                        top_words.append(processed_word)
                        seen_words.add(processed_word)
                        
                self.topic_word_lists[topic_id] = top_words
            else:
                # Assume it's already a list of words
                word_list = list(word_scores)[:self.top_words] if word_scores else []
                # Clean and deduplicate
                processed_words = []
                seen_words = set()
                for word in word_list:
                    processed_word = str(word).strip().lower() if word else ""
                    if processed_word and processed_word not in seen_words:
                        processed_words.append(processed_word)
                        seen_words.add(processed_word)
                self.topic_word_lists[topic_id] = processed_words
            
            # Add to overall vocabulary (deduplicated)
            if self.topic_word_lists.get(topic_id):
                self.all_words.update(self.topic_word_lists[topic_id])
        
        # Validate that we have topics with words
        if not self.topic_word_lists:
            print("Warning: No valid topics with words found after preprocessing")
        elif len(self.topic_word_lists) == 1:
            print("Warning: Only one topic found - diversity calculations may not be meaningful")
    
    def calculate_proportion_unique_words(self):
        """
        Calculate Proportion of Unique Words (PUW) - Intra-topic diversity
        
        Returns:
            float: Ratio of unique words across all topics vs total words
        """
        if not self.topic_word_lists:
            print("Warning: No topic word lists available for PUW calculation")
            return 0.0
        
        # Count total word instances across all topics (allowing duplicates across topics)
        total_word_instances = sum(len(words) for words in self.topic_word_lists.values())
        
        # Count unique words across all topics
        unique_words = len(self.all_words)
        
        if total_word_instances == 0:
            print("Warning: No words found in any topic for PUW calculation")
            return 0.0
        
        if unique_words == 0:
            print("Warning: No unique words found for PUW calculation")
            return 0.0
            
        puw_score = unique_words / total_word_instances
        
        # PUW should be between 0 and 1, with higher values indicating more diversity
        return min(1.0, max(0.0, puw_score))
    
    def calculate_jaccard_similarity(self, topic1_words, topic2_words):
        """
        Calculate Jaccard similarity between two topic word lists
        
        Args:
            topic1_words (list): Words from first topic
            topic2_words (list): Words from second topic
            
        Returns:
            float: Jaccard similarity (0-1)
        """
        if not isinstance(topic1_words, (list, set)) or not isinstance(topic2_words, (list, set)):
            print("Warning: Invalid input types for Jaccard similarity calculation")
            return 0.0
            
        set1 = set(topic1_words) if topic1_words else set()
        set2 = set(topic2_words) if topic2_words else set()
        
        # Handle edge cases
        if len(set1) == 0 and len(set2) == 0:
            return 1.0  # Both empty sets are considered identical
        
        if len(set1) == 0 or len(set2) == 0:
            return 0.0  # One empty set means no similarity
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        jaccard_sim = intersection / union
        return min(1.0, max(0.0, jaccard_sim))  # Ensure result is in [0, 1]
    
    def calculate_cosine_similarity(self, topic1_words, topic2_words):
        """
        Calculate cosine similarity between two topic word lists using binary vectors
        
        Args:
            topic1_words (list): Words from first topic
            topic2_words (list): Words from second topic
            
        Returns:
            float: Cosine similarity (0-1)
        """
        if not isinstance(topic1_words, (list, set)) or not isinstance(topic2_words, (list, set)):
            print("Warning: Invalid input types for cosine similarity calculation")
            return 0.0
            
        set1 = set(topic1_words) if topic1_words else set()
        set2 = set(topic2_words) if topic2_words else set()
        
        # Handle edge cases
        if len(set1) == 0 and len(set2) == 0:
            return 1.0  # Both empty sets are considered identical
        
        if len(set1) == 0 or len(set2) == 0:
            return 0.0  # One empty set means no similarity
        
        # Create binary vectors based on vocabulary
        vocab = list(self.all_words)
        if len(vocab) == 0:
            print("Warning: No vocabulary available for cosine similarity calculation")
            return 0.0
        
        # Optimized binary vector creation
        vector1 = np.zeros(len(vocab), dtype=np.float32)
        vector2 = np.zeros(len(vocab), dtype=np.float32)
        
        vocab_dict = {word: idx for idx, word in enumerate(vocab)}
        
        for word in set1:
            if word in vocab_dict:
                vector1[vocab_dict[word]] = 1.0
                
        for word in set2:
            if word in vocab_dict:
                vector2[vocab_dict[word]] = 1.0
        
        # Calculate cosine similarity
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        return min(1.0, max(0.0, cosine_sim))  # Ensure result is in [0, 1]
    
    def calculate_pairwise_similarities(self, similarity_func):
        """
        Calculate pairwise similarities between all topic pairs
        
        Args:
            similarity_func: Function to calculate similarity between two topic word lists
            
        Returns:
            dict: Dictionary of pairwise similarities
        """
        if not callable(similarity_func):
            raise ValueError("similarity_func must be callable")
            
        if not self.topic_word_lists:
            print("Warning: No topic word lists available for pairwise similarity calculation")
            return {}
        
        similarities = {}
        topic_ids = list(self.topic_word_lists.keys())
        
        if len(topic_ids) < 2:
            print("Warning: Need at least 2 topics for pairwise similarity calculation")
            return {}
        
        for i in range(len(topic_ids)):
            for j in range(i + 1, len(topic_ids)):
                topic1_id = topic_ids[i]
                topic2_id = topic_ids[j]
                
                try:
                    similarity = similarity_func(
                        self.topic_word_lists[topic1_id],
                        self.topic_word_lists[topic2_id]
                    )
                    
                    # Validate similarity score - only check for truly invalid values
                    if not isinstance(similarity, (int, float)) or math.isnan(similarity) or math.isinf(similarity):
                        print(f"Warning: Invalid similarity score for {topic1_id} vs {topic2_id}: {similarity}")
                        similarity = 0.0
                    else:
                        # Ensure similarity is in valid range [0, 1] - clamp silently for floating point precision
                        similarity = max(0.0, min(1.0, float(similarity)))
                    
                    pair_name = f"{topic1_id}_vs_{topic2_id}"
                    similarities[pair_name] = float(similarity)
                    
                except Exception as e:
                    print(f"Error calculating similarity for {topic1_id} vs {topic2_id}: {str(e)}")
                    pair_name = f"{topic1_id}_vs_{topic2_id}"
                    similarities[pair_name] = 0.0
        
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
            print("Warning: No similarities calculated for diversity calculation")
            return 0.0
        
        # Convert similarities to diversities (1 - similarity)
        diversities = []
        for sim in similarities.values():
            if isinstance(sim, (int, float)) and not math.isnan(sim):
                diversity = 1.0 - sim
                diversities.append(max(0.0, min(1.0, diversity)))  # Clamp to [0, 1]
            else:
                print(f"Warning: Invalid similarity value encountered: {sim}")
                diversities.append(0.0)
        
        if not diversities:
            return 0.0
            
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
        # Initialize default return structure
        default_result = {
            "proportion_unique_words": 0.0,
            "average_jaccard_diversity": 0.0,
            "average_cosine_diversity": 0.0,
            "jaccard_similarities": {},
            "cosine_similarities": {},
            "diversity_summary": {
                "most_similar_topics": [],
                "most_diverse_topics": [],
                "overall_diversity_score": 0.0,
                "total_topics": 0,
                "total_unique_words": 0,
                "calculation_warnings": []
            }
        }
        
        if not self.topic_word_lists:
            default_result["diversity_summary"]["calculation_warnings"].append("No topic word lists available")
            return default_result
        
        warnings = []
        
        # Update basic stats
        default_result["diversity_summary"]["total_topics"] = len(self.topic_word_lists)
        default_result["diversity_summary"]["total_unique_words"] = len(self.all_words)
        
        # Calculate intra-topic diversity (PUW)
        try:
            puw = self.calculate_proportion_unique_words()
            default_result["proportion_unique_words"] = puw
        except Exception as e:
            warnings.append(f"PUW calculation failed: {str(e)}")
            puw = 0.0
        
        # Calculate pairwise diversities
        try:
            jaccard_diversity = self.calculate_average_pairwise_diversity(self.calculate_jaccard_similarity)
            default_result["average_jaccard_diversity"] = jaccard_diversity
        except Exception as e:
            warnings.append(f"Jaccard diversity calculation failed: {str(e)}")
            jaccard_diversity = 0.0
        
        try:
            cosine_diversity = self.calculate_average_pairwise_diversity(self.calculate_cosine_similarity)
            default_result["average_cosine_diversity"] = cosine_diversity
        except Exception as e:
            warnings.append(f"Cosine diversity calculation failed: {str(e)}")
            cosine_diversity = 0.0
        
        # Get pairwise similarities for detailed analysis
        try:
            jaccard_similarities = self.calculate_pairwise_similarities(self.calculate_jaccard_similarity)
            default_result["jaccard_similarities"] = jaccard_similarities
        except Exception as e:
            warnings.append(f"Jaccard similarities calculation failed: {str(e)}")
            jaccard_similarities = {}
        
        try:
            cosine_similarities = self.calculate_pairwise_similarities(self.calculate_cosine_similarity)
            default_result["cosine_similarities"] = cosine_similarities
        except Exception as e:
            warnings.append(f"Cosine similarities calculation failed: {str(e)}")
            cosine_similarities = {}
        
        # Find most similar and diverse topic pairs
        try:
            most_similar = self.find_most_similar_topics(self.calculate_jaccard_similarity, top_k=3)
            default_result["diversity_summary"]["most_similar_topics"] = [pair[0] for pair in most_similar]
        except Exception as e:
            warnings.append(f"Most similar topics calculation failed: {str(e)}")
        
        try:
            most_diverse = self.find_most_diverse_topics(self.calculate_jaccard_similarity, top_k=3)
            default_result["diversity_summary"]["most_diverse_topics"] = [pair[0] for pair in most_diverse]
        except Exception as e:
            warnings.append(f"Most diverse topics calculation failed: {str(e)}")
        
        # Calculate overall diversity score with weighted average
        # PUW measures intra-topic diversity, while pairwise measures inter-topic diversity
        # Weight them appropriately: 40% PUW, 30% Jaccard, 30% Cosine
        valid_scores = []
        weights = []
        
        if puw > 0:
            valid_scores.append(puw)
            weights.append(0.4)
        
        if jaccard_diversity > 0:
            valid_scores.append(jaccard_diversity)
            weights.append(0.3)
            
        if cosine_diversity > 0:
            valid_scores.append(cosine_diversity)
            weights.append(0.3)
        
        if valid_scores and weights:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            overall_diversity = sum(score * weight for score, weight in zip(valid_scores, normalized_weights))
        else:
            overall_diversity = 0.0
            warnings.append("No valid diversity scores for overall calculation")
        
        default_result["diversity_summary"]["overall_diversity_score"] = min(1.0, max(0.0, overall_diversity))
        default_result["diversity_summary"]["calculation_warnings"] = warnings
        
        return default_result


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


def calculate_coherence_scores(topic_word_scores, output_dir=None, table_name=None, column_name=None, cleaned_data=None, topic_word_matrix=None, doc_topic_matrix=None, vocabulary=None, tokenizer=None, emoji_map=None,s_matrix = None):
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


    # Calculate LDAvis-style relevance scores if matrices are available
    print("Calculating LDAvis-style relevance scores...")

    relevance_scores = calculate_relevance_scores_from_matrix(
        topic_word_matrix=topic_word_matrix,
        doc_topic_matrix=doc_topic_matrix,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        emoji_map=emoji_map,
        s_matrix=s_matrix
    )
    
    if relevance_scores is not None:
        results["relevance"] = relevance_scores
        print(f"LDAvis-style relevance scores calculated for {len(relevance_scores)} topics")
    else:
        print("Warning: Could not calculate relevance scores.")
    # Calculate Diversity scores
    diversity_scores = calculate_diversity_scores(
        topic_word_matrix=topic_word_matrix, 
        vocabulary=vocabulary,
        top_words=50,
        output_dir=output_dir,
        table_name=table_name
    )

    if diversity_scores is not None:
        results["diversity"] = diversity_scores
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
                coherence="c_v",
                topn=15
            )
            umass_gensim = gensim_results.get_coherence()
            umass_per_topic = gensim_results.get_coherence_per_topic()
            # Create dictionary with topic-specific coherence scores
            topic_coherence_dict = {}
            for i, score in enumerate(umass_per_topic):
                topic_coherence_dict[f"Topic {i+1}"] = score.tolist() if hasattr(score, 'tolist') else score

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
        with open(coherence_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
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




def calculate_reconstruction_error(original_matrix, W_matrix, H_matrix):
    """
    Calculate reconstruction error metrics for NMF decomposition
    
    Args:
        original_matrix: Original TF-IDF sparse matrix (X)
        W_matrix: Document-topic matrix (W) from NMF - shape: (n_docs, n_topics) 
        H_matrix: Topic-word matrix (H) from NMF - shape: (n_topics, n_words)
        
    Returns:
        dict: Dictionary containing various reconstruction error metrics
    """
    if original_matrix is None or W_matrix is None or H_matrix is None:
        return {
            "frobenius_norm_squared": None,
            "mean_squared_error": None,
            "relative_error": None,
            "explained_variance_ratio": None,
            "sparsity_original": None,
            "sparsity_reconstructed": None
        }
    
    try:
        # Convert matrices to dense if they are sparse for calculation
        if hasattr(original_matrix, 'toarray'):
            X = original_matrix.toarray()
        else:
            X = np.array(original_matrix)
            
        W = np.array(W_matrix)
        H = np.array(H_matrix)
        
        # Reconstruct the matrix: X_reconstructed = W @ H
        X_reconstructed = W @ H
        
        # Calculate reconstruction error (difference matrix)
        error_matrix = X - X_reconstructed
        
        # 1. Frobenius norm squared: ||X - WH||_F^2
        frobenius_norm_squared = np.sum(error_matrix ** 2)
        
        # 2. Mean Squared Error: MSE = ||X - WH||_F^2 / (m * n)
        m, n = X.shape
        mean_squared_error = frobenius_norm_squared / (m * n)
        
        # 3. Relative Error: ||X - WH||_F / ||X||_F
        original_frobenius = np.sqrt(np.sum(X ** 2))
        if original_frobenius > 0:
            relative_error = np.sqrt(frobenius_norm_squared) / original_frobenius
        else:
            relative_error = 0.0
        
        # 4. Explained Variance Ratio: 1 - (||X - WH||_F^2 / ||X||_F^2)  
        original_frobenius_squared = np.sum(X ** 2)
        if original_frobenius_squared > 0:
            explained_variance_ratio = 1.0 - (frobenius_norm_squared / original_frobenius_squared)
        else:
            explained_variance_ratio = 1.0
        
        # 5. Sparsity metrics (percentage of zeros)
        total_elements = m * n
        sparsity_original = np.sum(X == 0) / total_elements
        sparsity_reconstructed = np.sum(X_reconstructed == 0) / total_elements
        
        return {
            "frobenius_norm_squared": float(frobenius_norm_squared),
            "mean_squared_error": float(mean_squared_error),
            "relative_error": float(relative_error),
            "explained_variance_ratio": float(explained_variance_ratio),
            "sparsity_original": float(sparsity_original),
            "sparsity_reconstructed": float(sparsity_reconstructed)
        }
        
    except Exception as e:
        print(f"Error calculating reconstruction error: {str(e)}")
        return {
            "frobenius_norm_squared": None,
            "mean_squared_error": None,
            "relative_error": None,
            "explained_variance_ratio": None,
            "sparsity_original": None,
            "sparsity_reconstructed": None
        }


def extract_topic_words_from_matrix(topic_word_matrix, vocabulary, top_n=50):
    """
    Extract top N words for each topic directly from the H matrix (topic-word matrix)
    
    Args:
        topic_word_matrix (numpy.ndarray): Topic-word probability matrix (H matrix from NMF) - shape: (n_topics, n_words)
        vocabulary (list): List of words corresponding to matrix columns
        top_n (int): Number of top words to extract per topic
        
    Returns:
        dict: Dictionary with topic_id as key and dict of {word: score} as value, or None if extraction fails
    """
    # Input validation
    if topic_word_matrix is None:
        print("Error: topic_word_matrix is None")
        return None
        
    if vocabulary is None:
        print("Error: vocabulary is None")
        return None
    
    if not isinstance(top_n, int) or top_n <= 0:
        print(f"Error: top_n must be a positive integer, got {top_n}")
        return None
    
    try:
        # Get matrix dimensions
        if hasattr(topic_word_matrix, 'shape'):
            n_topics, n_words = topic_word_matrix.shape
        else:
            print("Error: topic_word_matrix must have a shape attribute")
            return None
        
        if n_topics == 0 or n_words == 0:
            print(f"Error: Invalid matrix dimensions: ({n_topics}, {n_words})")
            return None
        
        # Validate vocabulary
        if not isinstance(vocabulary, (list, tuple)):
            print("Error: vocabulary must be a list or tuple")
            return None
            
        if len(vocabulary) == 0:
            print("Error: vocabulary is empty")
            return None
        
        # Handle vocabulary size mismatch
        if len(vocabulary) != n_words:
            print(f"Warning: Vocabulary size ({len(vocabulary)}) doesn't match matrix columns ({n_words})")
            min_size = min(len(vocabulary), n_words)
            vocabulary = list(vocabulary)[:min_size]
            if hasattr(topic_word_matrix, 'toarray'):
                # Handle sparse matrices
                topic_word_matrix = topic_word_matrix.toarray()[:, :min_size]
            else:
                topic_word_matrix = topic_word_matrix[:, :min_size]
            n_words = min_size
        
        topic_word_scores = {}
        
        for topic_idx in range(n_topics):
            try:
                topic_scores = topic_word_matrix[topic_idx]
                
                # Handle sparse matrices
                if hasattr(topic_scores, 'toarray'):
                    topic_scores = topic_scores.toarray().flatten()
                
                # Validate topic scores
                if len(topic_scores) != n_words:
                    print(f"Warning: Topic {topic_idx} scores length ({len(topic_scores)}) doesn't match vocabulary size ({n_words})")
                    continue
                
                # Get top N words for this topic
                if np.all(topic_scores == 0):
                    print(f"Warning: Topic {topic_idx} has all zero scores")
                    # Still include it but with empty scores
                    topic_word_dict = {}
                else:
                    # Get indices of top N words
                    top_indices = np.argsort(topic_scores)[::-1][:top_n]
                    
                    topic_word_dict = {}
                    for word_idx in top_indices:
                        if 0 <= word_idx < len(vocabulary):
                            word = vocabulary[word_idx]
                            score = float(topic_scores[word_idx])
                            
                            # Only include words with positive scores
                            if score > 0 and word and isinstance(word, str):
                                topic_word_dict[word.strip()] = score
                
                topic_id = f"topic_{topic_idx}"
                topic_word_scores[topic_id] = topic_word_dict
                
            except Exception as e:
                print(f"Error processing topic {topic_idx}: {str(e)}")
                # Continue with other topics
                continue
        
        if not topic_word_scores:
            print("Error: No valid topics extracted from matrix")
            return None
        
        print(f"Successfully extracted {len(topic_word_scores)} topics from matrix")
        return topic_word_scores
        
    except Exception as e:
        print(f"Error in extract_topic_words_from_matrix: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def calculate_diversity_scores(topics_json=None, topic_word_matrix=None, vocabulary=None, top_words=50, output_dir=None, table_name=None):
    """
    Calculate topic diversity scores from either topic word scores dictionary or H matrix
    
    Args:
        topics_json (dict, optional): Dictionary containing topics and their word scores
        topic_word_matrix (numpy.ndarray, optional): H matrix - shape: (n_topics, n_words)
        vocabulary (list, optional): Vocabulary list corresponding to matrix columns (required if using matrix)
        top_words (int): Number of top words to consider per topic (default: 50)
        output_dir (str, optional): Directory to save results
        table_name (str, optional): Name for output file
        
    Returns:
        dict: Dictionary containing all diversity metrics, or None if calculation fails
    """
    print("Calculating topic diversity scores...")
    
    # Determine input source
    if topics_json is not None:
        print(f"Using provided topic word scores dictionary with {len(topics_json)} topics")
        topic_word_scores = topics_json
    elif topic_word_matrix is not None and vocabulary is not None:
        print("Extracting topic word scores from H matrix")
        topic_word_scores = calculate_diversity_scores_from_matrix(
            topic_word_matrix, vocabulary, top_words
        )
        if topic_word_scores is None:
            return None
    else:
        print("Error: Either topics_json or (topic_word_matrix + vocabulary) must be provided")
        return None
    
    try:
        # Initialize TopicDiversityScorer
        diversity_calc = TopicDiversityScorer(topic_word_scores, top_words=top_words)
        
        # Calculate all diversity metrics
        diversity_results = diversity_calc.calculate_all_diversity_metrics()
        
        if diversity_results is None:
            print("Error: Diversity calculation returned None")
            return None
        
        # Add calculation metadata
        if topic_word_matrix is not None:
            diversity_results["calculation_metadata"] = {
                "input_source": "matrix",
                "matrix_shape": topic_word_matrix.shape,
                "vocabulary_size": len(vocabulary),
                "top_words_used": top_words,
                "topics_processed": len(topic_word_scores)
            }
        else:
            diversity_results["calculation_metadata"] = {
                "input_source": "topics_dict",
                "topics_provided": len(topics_json),
                "top_words_used": top_words,
                "topics_processed": len(topic_word_scores)
            }
        
        # Print summary
        overall_score = diversity_results['diversity_summary']['overall_diversity_score']
        puw_score = diversity_results['proportion_unique_words']
        jaccard_div = diversity_results['average_jaccard_diversity']
        cosine_div = diversity_results['average_cosine_diversity']
        
        print(f"Diversity Calculation Results:")
        print(f"  Overall Diversity Score: {overall_score:.4f}")
        print(f"  Proportion Unique Words: {puw_score:.4f}")
        print(f"  Average Jaccard Diversity: {jaccard_div:.4f}")
        print(f"  Average Cosine Diversity: {cosine_div:.4f}")
        
        # Save results if output directory is specified
        if output_dir and table_name:
            output_path = Path(output_dir)
            diversity_file = output_path / f"{table_name}_diversity_scores.json"
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(diversity_file, "w", encoding="utf-8") as f:
                json.dump(diversity_results, f, indent=4, ensure_ascii=False)
            print(f"Diversity scores saved to: {diversity_file}")
        
        return diversity_results
        
    except Exception as e:
        print(f"Error during diversity calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def calculate_diversity_scores_from_matrix(topic_word_matrix, vocabulary, top_words=50):
    """
    Calculate topic diversity scores directly from H matrix (topic-word matrix)
    
    Args:
        topic_word_matrix (numpy.ndarray): H matrix - shape: (n_topics, n_words)
        vocabulary (list): Vocabulary list corresponding to matrix columns
        top_words (int): Number of top words to consider per topic
        
    Returns:
        dict: Dictionary containing all diversity metrics, or None if calculation fails
    """
    # Input validation
    if topic_word_matrix is None:
        print("Error: topic_word_matrix is required for diversity calculation.")
        return None
        
    if vocabulary is None:
        print("Error: vocabulary is required for diversity calculation.")
        return None
    
    if not isinstance(top_words, int) or top_words <= 0:
        print(f"Error: top_words must be a positive integer, got {top_words}")
        return None
    
    # Validate matrix dimensions
    try:
        if hasattr(topic_word_matrix, 'shape'):
            n_topics, n_words = topic_word_matrix.shape
            if n_topics == 0 or n_words == 0:
                print(f"Error: Invalid matrix dimensions: {topic_word_matrix.shape}")
                return None
        else:
            print("Error: topic_word_matrix must have a shape attribute")
            return None
    except Exception as e:
        print(f"Error: Could not get matrix dimensions: {str(e)}")
        return None
    
    # Validate vocabulary
    if not isinstance(vocabulary, (list, tuple)) or len(vocabulary) == 0:
        print("Error: vocabulary must be a non-empty list or tuple")
        return None
    
    if len(vocabulary) != n_words:
        print(f"Warning: Vocabulary size ({len(vocabulary)}) doesn't match matrix columns ({n_words})")
        # Truncate vocabulary to match matrix if needed
        vocabulary = list(vocabulary)[:n_words]
    
    print(f"Extracting topics from H matrix: {n_topics} topics, {n_words} words, top {top_words} words per topic")
    
    try:
        # Extract words directly from H matrix
        topic_word_scores = extract_topic_words_from_matrix(
            topic_word_matrix, vocabulary, top_n=top_words
        )
        
        if topic_word_scores is None:
            print("Error: extract_topic_words_from_matrix returned None")
            return None
            
        if len(topic_word_scores) == 0:
            print("Error: No topics extracted from H matrix")
            return None
        
        print(f"Successfully extracted {len(topic_word_scores)} topics from matrix")
        return topic_word_scores
        
    except Exception as e:
        print(f"Error during topic extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# Alias for convenience - same function with clearer name
def calculate_topic_diversity(H_matrix, vocabulary, top_words=50):
    """
    Convenience function: Calculate topic diversity directly from NMF H matrix
    
    Args:
        H_matrix (numpy.ndarray): NMF H matrix (topic-word matrix) - shape: (n_topics, n_words)
        vocabulary (list): Vocabulary list from TF-IDF vectorizer
        top_words (int): Number of top words to consider per topic (default: 50)
        
    Returns:
        dict: Dictionary containing all diversity metrics including:
            - proportion_unique_words: Ratio of unique words across topics
            - average_jaccard_diversity: Average Jaccard-based diversity
            - average_cosine_diversity: Average cosine-based diversity
            - diversity_summary: Overall analysis with most similar/diverse pairs
    
    Example:
        >>> diversity_results = calculate_topic_diversity(H_matrix, vocabulary, top_words=50)
        >>> print(f"Overall diversity: {diversity_results['diversity_summary']['overall_diversity_score']:.4f}")
    """
    return calculate_diversity_scores(
        topic_word_matrix=H_matrix, 
        vocabulary=vocabulary, 
        top_words=top_words
    )


def calculate_relevance_scores_from_matrix(topic_word_matrix, doc_topic_matrix, vocabulary, tokenizer=None, emoji_map=None, s_matrix=None):
    """
    Calculate LDAvis-style relevance scores directly from NMF matrices

    Args:
        topic_word_matrix (numpy.ndarray): H matrix - shape: (n_topics, n_words)
        doc_topic_matrix (numpy.ndarray): W matrix - shape: (n_docs, n_topics)
        vocabulary (list): Vocabulary list corresponding to matrix columns
        tokenizer: Optional tokenizer for vocabulary creation
        emoji_map: Optional emoji map for decoding
        s_matrix (numpy.ndarray): S matrix for NMTF - shape: (k1, k2). Optional, used for NMTF factorization.

    Returns:
        dict: Dictionary containing all relevance scores and metadata
    """
    if topic_word_matrix is None or doc_topic_matrix is None:
        print("Error: Both topic_word_matrix and doc_topic_matrix are required for relevance calculation.")
        return None
    
    if vocabulary is None and tokenizer is None:
        print("Error: Either vocabulary or tokenizer must be provided for relevance calculation.")
        return None
    
    print(f"Calculating LDAvis relevance directly from matrices using vocabulary size: {len(vocabulary) if vocabulary else 'from tokenizer'}")
    
    try:
        # Calculate relevance scores using the new scorer
        relevance_results = get_topic_top_terms(
            h_matrix=topic_word_matrix,
            w_matrix=doc_topic_matrix,
            vocab=vocabulary,
            tokenizer=tokenizer,
            emoji_map=emoji_map,
            lambda_val=0.6,  # Standard lambda range
            top_n=30,  # Standard top N terms
            s_matrix=s_matrix
        )

        return relevance_results
        
    except Exception as e:
        print(f"Error calculating relevance scores: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


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


