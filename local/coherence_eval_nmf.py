import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from functions.nmf import run_nmf
from functions.turkish.turkish_preprocessor import metin_temizle
from functions.turkish.turkish_tokenizer_factory import init_tokenizer, train_tokenizer
from functions.turkish.turkish_text_encoder import veri_sayisallastir
from functions.tfidf import tf_idf_generator, tfidf_hesapla

from functions.english.english_vocabulary import sozluk_yarat
from functions.english.english_preprocessor import preprocess
COHERENCE_TYPE = "c_v"  # Default coherence type for Gensim CoherenceModel

def process_turkish_file(df,desired_columns: str, tokenizer=None, tokenizer_type="bpe"):
    
    metin_array = metin_temizle(df, desired_columns)
    print(f"Number of documents: {len(metin_array)}")

    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = init_tokenizer(tokenizer_type=tokenizer_type)

    # Train the tokenizer
    tokenizer = train_tokenizer(tokenizer, metin_array, tokenizer_type=tokenizer_type)
    sozluk = list(tokenizer.get_vocab().keys())

    # sayısallaştır
    sayisal_veri = veri_sayisallastir(metin_array, tokenizer)
    tdm = tf_idf_generator(sayisal_veri, tokenizer)

    return tdm, sozluk, sayisal_veri, tokenizer



def process_english_file(df,desired_columns: str, lemmatize: bool):
    sozluk, N = sozluk_yarat(df, desired_columns, lemmatize=LEMMATIZE)
    sayisal_veri = tfidf_hesapla(N, sozluk=sozluk, data=df, alanadi=desired_columns, output_dir=None, lemmatize=LEMMATIZE)
    tdm = sayisal_veri

    return tdm, sozluk, sayisal_veri


def print_topics(H, sozluk, top_n=15):
    topics = []
    # Create reverse mapping from index to word
    if isinstance(sozluk, dict):
        idx2word = {v: k for k, v in sozluk.items()}
    elif isinstance(sozluk, list):
        idx2word = {i: word for i, word in enumerate(sozluk)}
    else:
        raise TypeError(
            f"Unsupported type for sozluk: {type(sozluk)}. Expected list or dict."
        )

    for topic_idx, topic in enumerate(H):
        # Get indices and scores of top N words for this topic
        top_indices = topic.argsort()[: -top_n - 1 : -1]
        top_scores = topic[top_indices]
        # Map indices to words and pair with scores
        topic_words = [
            (idx2word[i], score)
            for i, score in zip(top_indices, top_scores)
            if i in idx2word
        ]
        # Sort by score in descending order
        topic_words = sorted(topic_words, key=lambda x: x[1], reverse=True)
        # Extract just the words
        topic_words = [word for word, _ in topic_words[:top_n]]
        topics.append(topic_words)
        print(f"Topic {topic_idx + 1}: {', '.join(topic_words)}")
    return topics


def get_topics_from_nmf(H, sozluk, top_n=15):
    """Extracts top N words for each topic from the NMF H matrix."""
    topics = []
    # Create reverse mapping from index to word
    # Check if sozluk is a list or dict (handle both cases for robustness, though user indicates list)
    if isinstance(sozluk, dict):
        # Original handling for dictionary
        idx2word = {v: k for k, v in sozluk.items()}
    elif isinstance(sozluk, list):
        # Handle list: assume index is ID, value is word
        idx2word = {i: word for i, word in enumerate(sozluk)}
    else:
        raise TypeError(
            f"Unsupported type for sozluk: {type(sozluk)}. Expected list or dict."
        )

    for topic_idx, topic in enumerate(H):
        # Get indices of top N words for this topic
        top_word_indices = topic.argsort()[: -len(sozluk) - 1 : -1]
        # Map indices to words
        topic_words = [idx2word[i] for i in top_word_indices if i in idx2word]
        topics.append(topic_words)
    return topics


def load_data(filepath: str, desired_columns: str) -> pd.DataFrame:
    """
    Load data from CSV or Excel file and perform initial preprocessing.

    Args:
        filepath (str): Path to the input file
        desired_columns (str): Column name containing the text to analyze

    Returns:
        pd.DataFrame: Loaded and preprocessed dataframe
    """
    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath, on_bad_lines="skip", encoding="utf-8", sep=";")
        else:
            df = pd.read_excel(filepath)

        if "COUNTRY" in df.columns:
            df = df[df["COUNTRY"] == "TR"]

        df = df.dropna(subset=[desired_columns])
        df = df.drop_duplicates()

        return df
    except Exception as e:
        print(f"Error during data loading: {e}")
        return None




def prepare_gensim_corpus(texts: list) -> tuple:
    """
    Prepare Gensim dictionary and corpus for coherence calculation.

    Args:
        texts (list): List of tokenized documents

    Returns:
        tuple: (gensim_dict, corpus)
    """
    gensim_dict = Dictionary(texts)
    corpus = [gensim_dict.doc2bow(text) for text in texts]
    print(f"Gensim dictionary size: {len(gensim_dict)}")
    return gensim_dict, corpus


def calculate_coherence_for_topics(
    topic_range: range, 
    tdm, 
    sozluk: list, 
    texts: list, 
    gensim_dict, 
    N_WORDS: int
) -> list:
    """
    Run NMF for each topic count and calculate coherence scores.

    Args:
        topic_range (range): Range of topic counts to evaluate
        tdm: Term-document matrix
        sozluk (list): Vocabulary list
        texts (list): List of tokenized documents
        gensim_dict: Gensim dictionary
        N_WORDS (int): Number of top words to consider per topic

    Returns:
        tuple: (coherence_values, last_H_matrix)
    """
    coherence_values = []
    last_H = None

    for num_topics in topic_range:
        print(f"  Testing {num_topics} topics...")
        try:
            # Run NMF
            W, H = run_nmf(
                num_of_topics=num_topics,
                sparse_matrix=tdm,
                norm_thresh=0.005,
                nmf_method="nmf"
            )
            last_H = H

            # Extract topics in the format needed by CoherenceModel
            topics = get_topics_from_nmf(H, sozluk, top_n=N_WORDS)
            if not topics or not any(topics):
                print(f"    Warning: No topics extracted for {num_topics} topics. Skipping coherence calculation.")
                coherence_values.append((num_topics, None))
                continue

            # Calculate Coherence Score
            coherence_model_cv = CoherenceModel(
                topics=topics,
                texts=texts,
                dictionary=gensim_dict,
                coherence=COHERENCE_TYPE,
                topn=N_WORDS,
            )
            coherence_cv = coherence_model_cv.get_coherence()
            coherence_values.append((num_topics, coherence_cv))
            print(f"    Coherence {COHERENCE_TYPE} for {num_topics} topics: {coherence_cv:.4f}")

        except Exception as e:
            print(f"    Error processing {num_topics} topics: {e}")
            coherence_values.append((num_topics, None))

    return coherence_values, last_H


def plot_and_save_results(
    coherence_values: list, 
    H, 
    sozluk: list, 
    N_WORDS: int, 
    desired_columns: str, 
    base_table_name: str
) -> list:
    """
    Plot coherence scores and save results.

    Args:
        coherence_values (list): List of tuples containing (topic_count, coherence_score)
        H: NMF H matrix for the last topic count
        sozluk (list): Vocabulary list
        N_WORDS (int): Number of top words to consider per topic
        desired_columns (str): Column name containing the text to analyze
        base_table_name (str): Base name for output files

    Returns:
        list: List of valid coherence scores
    """
    valid_scores = [(k, v) for k, v in coherence_values if v is not None]
    if not valid_scores:
        print("No valid coherence scores were calculated. Cannot plot.")
        return valid_scores

    # Create output directories
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "Output")
    table_output_dir = os.path.join(output_dir, base_table_name)
    os.makedirs(table_output_dir, exist_ok=True)

    # Save coherence scores to CSV file
    csv_filename = os.path.join(table_output_dir, "coherence_scores.csv")
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Number_of_Topics", "Coherence_Score"])  # Header
        writer.writerows(valid_scores)
    print(f"Coherence scores have been saved to: {csv_filename}")

    # Plot coherence scores
    topic_counts = [item[0] for item in valid_scores]
    scores = [item[1] for item in valid_scores]

    plt.figure(figsize=(10, 5))
    plt.plot(topic_counts, scores, marker="o")
    plt.xlabel("Number of Topics")
    plt.ylabel(f"Coherence Score {COHERENCE_TYPE}")
    plt.title(f"NMF Coherence Scores {COHERENCE_TYPE} for '{desired_columns}'")
    plt.xticks(np.arange(min(topic_counts), max(topic_counts) + 2, 1))
    plot_filename = os.path.join(table_output_dir, f"{base_table_name}_coherence_plot.png")
    plt.savefig(plot_filename)
    print(f"Coherence plot saved to: {plot_filename}")

    # Find the optimal number of topics (highest coherence)
    optimal_num_topics = topic_counts[scores.index(max(scores))]
    print("--- Optimal Number of Topics ---")
    print(f"Optimal number of topics based on c_v coherence: {optimal_num_topics} (Score: {max(scores):.4f})")
    print("--- Topics ---")
    print_topics(H, sozluk, N_WORDS)
    print("--- End of Coherence Evaluation ---")

    return valid_scores


def evaluate_nmf_coherence(
    filepath: str,
    desired_columns: str,
    topic_range: range,
    base_table_name: str = "evaluation_data",
    lemmatize: bool = False,
    N_WORDS: int = 15,
    tokenizer = None,
    LANGUAGE: str = "TR",
    
) -> list:
    """
    Runs NMF for a range of topic counts, calculates coherence, and plots results.
    Using language-specific text processing.

    Args:
        filepath (str): Path to the input CSV file
        desired_columns (str): Column name containing the text to analyze
        topic_range (range): Range of topic counts to evaluate
        base_table_name (str): Base name for output files
        lemmatize (bool): Whether to lemmatize text
        N_WORDS (int): Number of top words to consider per topic
        tokenizer: Pre-initialized tokenizer (if None, a new one will be created)
        LANGUAGE (str): Language of the text ("TR" for Turkish, "EN" for English)

    Returns:
        list: List of tuples containing (topic_count, coherence_score)
    """
    print(f"Starting coherence evaluation for topics in range {topic_range.start} to {topic_range.stop - 1}")

    # Setup output directories
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "Output")
    tfidf_dir = os.path.join(output_dir, "tfidf")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tfidf_dir, exist_ok=True)

    # 1. Load data
    print("Reading and preprocessing data...")
    df = load_data(filepath, desired_columns)
    if df is None:
        return []

    # 2. Process text based on language
    print("Cleaning and processing text...")
    try:
        if LANGUAGE == "TR":
            tokenizer = init_tokenizer(tokenizer_type=tokenizer_type) if tokenizer is None else tokenizer
            tdm, sozluk, sayisal_veri, tokenizer = process_turkish_file(df, desired_columns, tokenizer, tokenizer_type)
        elif LANGUAGE == "EN":
            tdm, sozluk, sayisal_veri = process_english_file(df, desired_columns, lemmatize)
        else:
            raise ValueError(f"Invalid language: {LANGUAGE}")

        print(f"Vocabulary size: {len(sozluk)}")
        texts = []
        for doc in sayisal_veri:
            words = [sozluk[token_id] for token_id in doc if token_id < len(sozluk)]
            texts.append(words)
                
        if not texts:
            raise ValueError("Text processing resulted in empty texts")

    except Exception as e:
        print(f"Error during text processing: {e}")
        return []

    # 3. Prepare Gensim corpus
    print("Preparing Gensim dictionary and corpus...")
    try:
        gensim_dict, corpus = prepare_gensim_corpus(texts)
        if not corpus:
            raise ValueError("Corpus creation resulted in an empty corpus")
    except Exception as e:
        print(f"Error during Gensim preparation: {e}")
        return []

    # 4. Calculate coherence for each topic count
    print("Running NMF and calculating coherence for different topic counts...")
    coherence_values, last_H = calculate_coherence_for_topics(
        topic_range, tdm, sozluk, texts, gensim_dict, N_WORDS
    )

    # 5. Plot and save results
    print("Plotting coherence scores...")
    valid_scores = plot_and_save_results(   
        coherence_values, last_H, sozluk, N_WORDS, desired_columns, base_table_name
    )

    return coherence_values


def run_coherence_evaluation(
    filepath: str,
    desired_columns: str,
    min_topics: int = 2,
    max_topics: int = 15,
    step: int = 2,
    base_name: str = "evaluation",
    N_WORDS: int = 15,
    LEMMATIZE: bool = False,
    LANGUAGE: str = "TR",
    tokenizer_type: str = "bpe"
) -> list:
    """
    Run the coherence evaluation process with the given parameters.
    """
    print("Starting coherence evaluation process...")

    topic_range = range(min_topics, max_topics + 1, step)
    return evaluate_nmf_coherence(
        filepath=filepath,
        desired_columns=desired_columns,
        topic_range=topic_range,
        base_table_name=base_name,
        N_WORDS=N_WORDS,
        lemmatize=LEMMATIZE,
        LANGUAGE=LANGUAGE
    )


if __name__ == "__main__":
    LANGUAGE = "TR"  # or "EN"
    LEMMATIZE = False
    filepath = "veri_setleri/playstore.csv"
    table_name = "PLAYSTORE_nmf"
    desired_columns = "REVIEW_TEXT"
    desired_topics = 14
    N_WORDS = 15
    tokenizer_type = "bpe"

    min_topics = 2
    max_topics = 15
    step = 1
    base_name = table_name +"_evaluation_" + COHERENCE_TYPE + "_" + tokenizer_type

    # Coherence evaluation example
    run_coherence_evaluation(
        filepath,
        desired_columns,
        min_topics,
        max_topics,
        step,
        base_name,
        N_WORDS,
        LEMMATIZE,
        LANGUAGE,
        tokenizer_type
    )
