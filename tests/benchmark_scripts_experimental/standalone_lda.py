#!/usr/bin/env python3
"""
Standalone LDA Implementation for Comparison with MANTA NMF

This script implements Latent Dirichlet Allocation (LDA) using the exact same preprocessing
pipeline and coherence evaluation methodology as the MANTA NMF system, enabling fair
comparison between LDA and NMF topic modeling approaches.

Features:
- Same text preprocessing (NLTK lemmatization/stemming, stopword removal)
- Identical TF-IDF vectorization (with BM25 and pivoted normalization options)
- Same coherence scoring (Gensim c_v metric)
- Compatible topic export format
- Coherence evaluation across topic ranges

Author: Generated for NMF-LDA comparison
"""

import os
import csv
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from wordcloud import WordCloud

# Import MANTA preprocessing functions
from manta._functions.english.english_preprocessor import clean_english_text
from manta._functions.english.english_vocabulary import create_english_vocab
from manta._functions.english.english_text_encoder import counterize_english
from manta._functions.tfidf.tfidf_english_calculator import tf_idf_english
from manta._functions.common_language.emoji_processor import EmojiMap
from manta.utils.analysis.coherence_score import calculate_coherence_scores

# Default coherence type (matches MANTA)
COHERENCE_TYPE = "c_v"


class StandaloneLDA:
    """
    Standalone LDA implementation using MANTA's preprocessing pipeline.
    
    This class provides LDA topic modeling with the exact same preprocessing
    and evaluation methodology as the MANTA NMF system.
    """
    
    def __init__(self, 
                 lemmatize: bool = False, 
                 emoji_map: EmojiMap = None,
                 use_bm25: bool = False,
                 use_pivoted_norm: bool = True,
                 random_state: int = 42):
        """
        Initialize the LDA processor.
        
        Args:
            lemmatize: Whether to apply lemmatization during preprocessing
            emoji_map: EmojiMap instance for emoji processing
            use_bm25: Whether to use BM25 instead of TF-IDF
            use_pivoted_norm: Whether to apply pivoted normalization
            random_state: Random seed for reproducible results
        """
        self.lemmatize = lemmatize
        self.emoji_map = emoji_map
        self.use_bm25 = use_bm25
        self.use_pivoted_norm = use_pivoted_norm
        self.random_state = random_state
        
        # Processed data storage
        self.cleaned_texts = None
        self.vocabulary = None
        self.vectorized_data = None
        self.tfidf_matrix = None
        self.lda_model = None
        
    def load_and_preprocess_data(self, filepath: str, desired_columns: str, separator: str = ";") -> pd.DataFrame:
        """
        Load and preprocess data using MANTA's approach.
        
        Args:
            filepath: Path to input CSV/Excel file
            desired_columns: Name of column containing text data
            
        Returns:
            Preprocessed DataFrame
        """
        print("Loading and preprocessing data...")
        
        # Load data (matches MANTA's approach)
        try:
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath, on_bad_lines="skip", encoding="utf-8", sep=separator)
            else:
                df = pd.read_excel(filepath)
                
            # Filter for Turkey if COUNTRY column exists (matches MANTA)
            if "COUNTRY" in df.columns:
                df = df[df["COUNTRY"] == "TR"]
                
            # Clean data (matches MANTA)
            df = df.dropna(subset=[desired_columns])
            df = df.drop_duplicates()
            
            print(f"Loaded {len(df)} documents after cleaning")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def process_english_text(self, df: pd.DataFrame, desired_columns: str) -> Tuple[Any, List[str], Any, List[str]]:
        """
        Process English text using MANTA's exact preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            desired_columns: Column name containing text
            
        Returns:
            Tuple of (tfidf_matrix, vocabulary, vectorized_data, cleaned_texts)
        """
        print("Processing English text using MANTA pipeline...")
        start_time = time.time()
        
        # Use MANTA's English preprocessing
        cleaned_texts = clean_english_text(
            metin=df[desired_columns], 
            lemmatize=self.lemmatize, 
            emoji_map=self.emoji_map
        )
        print(f"Text preprocessing completed in {time.time() - start_time:.2f} seconds")
        
        # Create vocabulary using MANTA's method
        vocabulary, N = create_english_vocab(cleaned_texts, desired_columns, lemmatize=self.lemmatize)
        
        # Vectorize using MANTA's method
        vectorized_data = counterize_english(
            N, vocab=vocabulary, data=cleaned_texts, 
            field_name=desired_columns, lemmatize=self.lemmatize
        )
        
        # Apply TF-IDF using MANTA's calculator
        tfidf_matrix = tf_idf_english(
            N=N, vocab=vocabulary, data=vectorized_data, 
            fieldname=desired_columns, output_dir=None,
            lemmatize=self.lemmatize,
            use_bm25=self.use_bm25,
            use_pivoted_norm=self.use_pivoted_norm
        )
        
        # Store processed data
        self.cleaned_texts = cleaned_texts
        self.vocabulary = vocabulary
        self.vectorized_data = vectorized_data
        self.tfidf_matrix = tfidf_matrix
        
        print(f"Vocabulary size: {len(vocabulary)}")
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        return tfidf_matrix, vocabulary, vectorized_data, cleaned_texts
    
    def fit_lda(self, num_topics: int, max_iter: int = 50) -> LatentDirichletAllocation:
        """
        Fit LDA model using scikit-learn.
        
        Args:
            num_topics: Number of topics to discover
            max_iter: Maximum iterations for LDA
            
        Returns:
            Fitted LDA model
        """
        print(f"Fitting LDA model with {num_topics} topics...")
        
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=max_iter,
            learning_method='online',
            learning_offset=50.0,
            random_state=self.random_state,
            n_jobs=1,  # Use single thread for consistency,
            verbose=1
        )
        
        # Fit on TF-IDF matrix
        lda.fit(self.tfidf_matrix)
        self.lda_model = lda
        
        print(f"LDA model fitted. Perplexity: {lda.perplexity(self.tfidf_matrix):.4f}")
        return lda
    
    def get_topic_words(self, num_words: int = 15) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract top words for each topic in MANTA's format.
        
        Args:
            num_words: Number of top words per topic
            
        Returns:
            Dictionary mapping topic names to word-score pairs
        """
        if self.lda_model is None:
            raise ValueError("LDA model not fitted yet")
            
        topics_data = {}
        
        # Get topic-word matrix
        components = self.lda_model.components_
        
        for topic_idx in range(self.lda_model.n_components):
            topic_name = f'Konu {topic_idx + 1:02d}'
            
            # Get word indices sorted by weight (descending)
            top_word_indices = components[topic_idx].argsort()[::-1][:num_words]
            
            # Create word:score pairs (matches MANTA format)
            topic_words = []
            for idx in top_word_indices:
                word = self.vocabulary[idx]
                # Remove "##" prefix if it exists (matches MANTA)
                if word.startswith("##"):
                    word = word[2:]
                score = float(components[topic_idx][idx])
                topic_words.append(f"{word}:{score:.8f}")
            
            topics_data[topic_name] = topic_words
        
        return topics_data
    
    def get_topics_for_coherence(self, num_words: int = 15) -> List[List[str]]:
        """
        Get topics in format needed by Gensim CoherenceModel.
        
        Args:
            num_words: Number of top words per topic
            
        Returns:
            List of topic word lists
        """
        if self.lda_model is None:
            raise ValueError("LDA model not fitted yet")
            
        topics_list = []
        components = self.lda_model.components_
        
        for topic_idx in range(self.lda_model.n_components):
            # Get top words for this topic
            top_word_indices = components[topic_idx].argsort()[::-1][:num_words]
            topic_words = [self.vocabulary[idx] for idx in top_word_indices]
            topics_list.append(topic_words)
        
        return topics_list
    
    def calculate_coherence(self, num_words: int = 15) -> float:
        """
        Calculate coherence score using Gensim (matches MANTA).
        
        Args:
            num_words: Number of top words to consider
            
        Returns:
            Coherence score
        """
        if self.lda_model is None:
            raise ValueError("LDA model not fitted yet")
            
        # Prepare data for Gensim coherence calculation
        texts = [doc.split() for doc in self.cleaned_texts]
        gensim_dict = Dictionary(texts)
        
        # Get topics in Gensim format
        topics = self.get_topics_for_coherence(num_words)
        
        # Calculate coherence using the same method as MANTA
        coherence_model = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=gensim_dict,
            coherence=COHERENCE_TYPE,
            topn=num_words,
        )
        
        return coherence_model.get_coherence()
    
    def print_topics(self, num_words: int = 15):
        """Print topics in a readable format."""
        topics_data = self.get_topic_words(num_words)
        
        for topic_name, words in topics_data.items():
            # Extract just the words (without scores) for display
            word_list = [word.split(":")[0] for word in words]
            print(f"{topic_name}: {', '.join(word_list)}")
    
    def generate_wordclouds(self, output_dir: str, base_name: str, num_words: int = 15):
        """
        Generate wordcloud images for each topic (matches MANTA functionality).
        
        Args:
            output_dir: Output directory for wordcloud images
            base_name: Base name for files
            num_words: Number of words per topic
        """
        topics_data = self.get_topic_words(num_words)
        
        # Create wordclouds directory
        wordcloud_dir = os.path.join(output_dir, "wordclouds")
        os.makedirs(wordcloud_dir, exist_ok=True)
        
        for topic_name, words in topics_data.items():
            # Extract words only (without scores)
            words_only = [word.split(":")[0] for word in words]
            
            # Generate wordcloud
            wordcloud = WordCloud(
                width=800, height=400, 
                background_color='white'
            ).generate(" ".join(words_only))
            
            # Save wordcloud
            filename = os.path.join(wordcloud_dir, f"{topic_name}_LDA.png")
            wordcloud.to_file(filename)
            
        print(f"Wordclouds saved to: {wordcloud_dir}")
    
    def save_topics_to_json(self, output_dir: str, base_name: str, num_words: int = 15):
        """
        Save topics to JSON file in MANTA format.
        
        Args:
            output_dir: Output directory
            base_name: Base name for files
            num_words: Number of words per topic
        """
        topics_data = self.get_topic_words(num_words)
        
        # Convert to format matching MANTA's output
        topic_word_scores = {}
        for topic_name, words in topics_data.items():
            word_scores = {}
            for word_score in words:
                word, score = word_score.split(":", 1)
                word_scores[word] = float(score)
            topic_word_scores[topic_name] = word_scores
        
        # Save to JSON
        json_filename = os.path.join(output_dir, f"{base_name}_LDA_topics.json")
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(topic_word_scores, f, ensure_ascii=False, indent=2)
        
        print(f"Topics saved to: {json_filename}")
        return topic_word_scores


def evaluate_lda_coherence(
    filepath: str,
    desired_columns: str,
    topic_range: range,
    base_table_name: str = "lda_evaluation",
    lemmatize: bool = False,
    num_words: int = 15,
    use_bm25: bool = False,
    use_pivoted_norm: bool = True
) -> List[Tuple[int, float]]:
    """
    Evaluate LDA coherence across different topic counts (matches MANTA's evaluation approach).
    
    Args:
        filepath: Path to input data file
        desired_columns: Column name containing text
        topic_range: Range of topic counts to evaluate
        base_table_name: Base name for output files
        lemmatize: Whether to apply lemmatization
        num_words: Number of top words to consider
        use_bm25: Whether to use BM25 instead of TF-IDF
        use_pivoted_norm: Whether to apply pivoted normalization
        
    Returns:
        List of (num_topics, coherence_score) tuples
    """
    print(f"Starting LDA coherence evaluation for topics in range {topic_range.start} to {topic_range.stop - 1}")
    
    # Setup output directories
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "Output")
    table_output_dir = os.path.join(output_dir, base_table_name)
    os.makedirs(table_output_dir, exist_ok=True)
    
    # Initialize LDA processor
    emoji_map = EmojiMap() if desired_columns else None
    lda_processor = StandaloneLDA(
        lemmatize=lemmatize, 
        emoji_map=emoji_map,
        use_bm25=use_bm25,
        use_pivoted_norm=use_pivoted_norm
    )
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = lda_processor.load_and_preprocess_data(filepath, desired_columns)
    if df is None:
        return []
    
    # Process text using MANTA pipeline
    print("Processing text using MANTA pipeline...")
    tfidf_matrix, vocabulary, vectorized_data, cleaned_texts = lda_processor.process_english_text(df, desired_columns)
    
    # Evaluate coherence for different topic counts
    coherence_values = []
    last_topics_data = None
    
    for num_topics in topic_range:
        print(f"  Testing {num_topics} topics...")
        try:
            # Fit LDA model
            lda_model = lda_processor.fit_lda(num_topics)
            
            # Calculate coherence
            coherence_score = lda_processor.calculate_coherence(num_words)
            coherence_values.append((num_topics, coherence_score))
            
            print(f"    Coherence {COHERENCE_TYPE} for {num_topics} topics: {coherence_score:.4f}")
            
            # Keep topics from last iteration for display
            if num_topics == max(topic_range):
                last_topics_data = lda_processor.get_topic_words(num_words)
                
        except Exception as e:
            print(f"    Error processing {num_topics} topics: {e}")
            coherence_values.append((num_topics, None))
    
    # Save and plot results (matches MANTA's approach)
    valid_scores = [(k, v) for k, v in coherence_values if v is not None]
    if not valid_scores:
        print("No valid coherence scores were calculated. Cannot plot.")
        return coherence_values
    
    # Save coherence scores to CSV
    csv_filename = os.path.join(table_output_dir, "lda_coherence_scores.csv")
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Number_of_Topics", "Coherence_Score"])
        writer.writerows(valid_scores)
    print(f"Coherence scores saved to: {csv_filename}")
    
    # Plot coherence scores
    topic_counts = [item[0] for item in valid_scores]
    scores = [item[1] for item in valid_scores]
    
    plt.figure(figsize=(10, 5))
    plt.plot(topic_counts, scores, marker="o", color="red", label="LDA")
    plt.xlabel("Number of Topics")
    plt.ylabel(f"Coherence Score {COHERENCE_TYPE}")
    plt.title(f"LDA Coherence Scores {COHERENCE_TYPE} for '{desired_columns}'")
    plt.xticks(np.arange(min(topic_counts), max(topic_counts) + 2, 1))
    plt.legend()
    plot_filename = os.path.join(table_output_dir, f"{base_table_name}_lda_coherence_plot.png")
    plt.savefig(plot_filename)
    print(f"Coherence plot saved to: {plot_filename}")
    
    # Find optimal number of topics
    optimal_num_topics = topic_counts[scores.index(max(scores))]
    print("--- Optimal Number of Topics ---")
    print(f"Optimal number of topics based on {COHERENCE_TYPE} coherence: {optimal_num_topics} (Score: {max(scores):.4f})")
    
    # Print topics from best model
    if last_topics_data:
        print("--- Topics ---")
        for topic_name, words in last_topics_data.items():
            word_list = [word.split(":")[0] for word in words]
            print(f"{topic_name}: {', '.join(word_list)}")
    
    print("--- End of LDA Coherence Evaluation ---")
    
    return coherence_values


def compare_lda_with_nmf(
    filepath: str,
    desired_columns: str,
    num_topics: int,
    base_name: str = "comparison",
    lemmatize: bool = False,
    num_words: int = 15,
    generate_wordclouds: bool = True,
    separator: str = ";"
) -> Dict[str, Any]:
    """
    Run LDA analysis and prepare for comparison with NMF.
    
    Args:
        filepath: Path to input data
        desired_columns: Column name containing text
        num_topics: Number of topics to discover
        base_name: Base name for output files
        lemmatize: Whether to apply lemmatization
        num_words: Number of words per topic
        generate_wordclouds: Whether to generate wordcloud images
        
    Returns:
        Dictionary with LDA results
    """
    print(f"Running LDA analysis with {num_topics} topics...")
    
    # Setup output directory
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "Output")
    table_output_dir = os.path.join(output_dir, f"{base_name}_LDA")
    os.makedirs(table_output_dir, exist_ok=True)
    
    # Initialize and run LDA
    emoji_map = EmojiMap() if desired_columns else None
    lda_processor = StandaloneLDA(lemmatize=lemmatize, emoji_map=emoji_map)
    
    try:
        # Load and process data
        df = lda_processor.load_and_preprocess_data(filepath, desired_columns, separator)
        if df is None:
            return {"error": "Failed to load data", "coherence_score": 0.0}
        
        lda_processor.process_english_text(df, desired_columns)
        
        # Fit LDA model
        lda_model = lda_processor.fit_lda(num_topics)
        
        # Calculate coherence
        coherence_score = lda_processor.calculate_coherence(num_words)
        
        # Get topics
        topics_data = lda_processor.get_topic_words(num_words)
        
        # Save results
        topic_word_scores = lda_processor.save_topics_to_json(table_output_dir, base_name, num_words)
        
        # Calculate and save coherence scores using MANTA's method
        coherence_scores = calculate_coherence_scores(
            topic_word_scores,
            output_dir=table_output_dir,
            column_name=desired_columns,
            cleaned_data=lda_processor.cleaned_texts,
            table_name=base_name
        )
        
        # Generate wordclouds if requested
        if generate_wordclouds:
            lda_processor.generate_wordclouds(table_output_dir, base_name, num_words)
        
        # Print results
        print(f"\nLDA Results ({num_topics} topics):")
        print(f"Coherence Score ({COHERENCE_TYPE}): {coherence_score:.4f}")
        print("\nTopics:")
        lda_processor.print_topics(num_words)
        
        return {
            "coherence_score": coherence_score,
            "topics_data": topics_data,
            "topic_word_scores": topic_word_scores,
            "model": lda_model,
            "output_dir": table_output_dir
        }
        
    except Exception as e:
        print(f"Error in LDA analysis: {str(e)}")
        return {
            "error": str(e),
            "coherence_score": 0.0,
            "topics_data": {},
            "topic_word_scores": {},
            "model": None,
            "output_dir": table_output_dir
        }


if __name__ == "__main__":
    # Example usage - modify these parameters as needed
    
    # Configuration (modify as needed)
    FILEPATH = "../../veri_setleri/playstore.csv"  # Path to your data file
    DESIRED_COLUMNS = "REVIEW_TEXT"           # Column containing text data
    BASE_NAME = "playstore_lda"               # Base name for output files
    LEMMATIZE = False                         # Whether to lemmatize text
    NUM_WORDS = 15                           # Number of words per topic
    
    # Coherence evaluation settings
    MIN_TOPICS = 2
    MAX_TOPICS = 15
    STEP = 1
    
    # Single analysis settings
    NUM_TOPICS = 5  # For single LDA run
    
    print("=== Standalone LDA Analysis ===")
    print(f"Using same preprocessing pipeline as MANTA NMF")
    print(f"Coherence metric: {COHERENCE_TYPE}")
    print(f"Lemmatization: {'Enabled' if LEMMATIZE else 'Disabled'}")
    
    # Option 1: Run coherence evaluation across topic range
    print("\n1. Running coherence evaluation across topic range...")
    topic_range = range(MIN_TOPICS, MAX_TOPICS + 1, STEP)
    coherence_results = evaluate_lda_coherence(
        filepath=FILEPATH,
        desired_columns=DESIRED_COLUMNS,
        topic_range=topic_range,
        base_table_name=f"{BASE_NAME}_evaluation",
        lemmatize=LEMMATIZE,
        num_words=NUM_WORDS
    )
    
    # Option 2: Run single LDA analysis
    print(f"\n2. Running single LDA analysis with {NUM_TOPICS} topics...")
    lda_results = compare_lda_with_nmf(
        filepath=FILEPATH,
        desired_columns=DESIRED_COLUMNS,
        num_topics=NUM_TOPICS,
        base_name=BASE_NAME,
        lemmatize=LEMMATIZE,
        num_words=NUM_WORDS,
        generate_wordclouds=True
    )
    
    print("\n=== Analysis Complete ===")
    print("Check the Output directory for:")
    print("- Coherence evaluation results and plots")
    print("- Topic wordclouds")
    print("- JSON files with topic-word scores")
    print("- CSV files with coherence scores")