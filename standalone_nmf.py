import os
import time

import pandas as pd
from sqlalchemy import create_engine

from functions.nmf import run_nmf
from functions.turkish.konuAnalizi import konu_analizi
from functions.turkish.sayisallastir import veri_sayisallastir
from functions.turkish.temizle import metin_temizle
from functions.tfidf import tf_idf_generator, tfidf_hesapla
from functions.turkish.token_yarat import init_tokenizer, train_tokenizer

from functions.english.sozluk import sozluk_yarat
from functions.english.process import preprocess

from utils.gen_cloud import generate_wordclouds
from utils.save_doc_score_pair import save_doc_score_pair
from utils.topic_dist import gen_topic_dist
from utils.word_cooccurrence import calc_word_cooccurrence
from utils.coherence_score import calculate_coherence_scores
from utils.export_excel import export_topics_to_excel

import numpy as np
import gensim

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
    sozluk, N = sozluk_yarat(df, desired_columns, lemmatize=lemmatize)
    sayisal_veri = tfidf_hesapla(N, sozluk=sozluk, data=df, alanadi=desired_columns, output_dir=None, lemmatize=lemmatize)
    tdm = sayisal_veri

    return tdm, sozluk, sayisal_veri



def process_file(
        filepath: str,
        table_name: str,
        desired_columns: str,
        desired_topic_count: int,
        LEMMATIZE: bool,
        N_TOPICS: int,
        tokenizer=None,
        LANGUAGE="TR",
        tokenizer_type="bpe",
        gen_topic_distribution=True,
        separator=",",
        nmf_type="nmf"
) -> dict:
    """
    Process a file and perform topic modeling without Celery/Redis dependencies

    Args:
        filepath (str): Path to the input CSV file
        table_name (str): Name of the table to store the data
        desired_columns (str): Column name to analyze
        desired_topic_count (int): Number of topics to extract
        LEMMATIZE (bool): Whether to lemmatize the text
        N_TOPICS (int): Number of words per topic
        tokenizer: Pre-initialized tokenizer (if None, a new one will be created)
        LANGUAGE (str): Language of the text ("TR" or "EN")
        tokenizer_type (str): Type of tokenizer to use for Turkish ("bpe" or "wordpiece")
        nmf_type (str): NMF method to use ("opnmf" or "nmf")
    Returns:
        dict: Result of the process containing state, message, and topic word scores dictionary
    """
    # Get base directory and create necessary directories
    base_dir = os.path.abspath(os.path.dirname(__file__))
    instance_path = os.path.join(base_dir, "instance")
    output_dir = os.path.join(base_dir, "Output")
    tfidf_dir = os.path.join(output_dir, "tfidf")

    # Create necessary directories first
    os.makedirs(instance_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tfidf_dir, exist_ok=True)

    # Database configurations using standard SQLAlchemy
    topics_db_eng = create_engine(
        f'sqlite:///{os.path.join(instance_path, "topics.db")}'
    )
    main_db_eng = create_engine(
        f'sqlite:///{os.path.join(instance_path, "scopus.db")}'
    )  # Main data DB

    try:
        print(f"Starting topic modeling for {table_name}")

        # Clean up the desired_columns
        desired_columns = desired_columns.strip() if desired_columns else None

        # Read the input file
        print("Reading input file...")
        # if file is csv, read it with read_csv
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath, on_bad_lines="skip", encoding="utf-8", sep=separator)
        else:
            df = pd.read_excel(filepath)

        # Add to main database

        # INSTEAD OF SAVING WHOLE TABLE TO DATABASE, SAVE ONLY THE DESIRED COLUMNS
        df = df[desired_columns]  # Use double brackets to select columns
        df = df.dropna()

        # remove duplicates
        count_of_duplicates = df.duplicated().sum()
        total_rows = len(df)
        if total_rows*0.9 < count_of_duplicates:
            print(f"Warning: {count_of_duplicates} duplicates found in the data, which is more than 90% of the total rows ({total_rows}).")
            df = df.drop_duplicates()
        df = df.drop_duplicates()
        print(f"File has {len(df)} rows.")

        print("Adding data to main database...")    
        # Check if table exists using a SQL query instead of direct table read
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        existing_tables = pd.read_sql_query(tables_query, main_db_eng)
        if table_name not in existing_tables['name'].tolist():
            # Use the main_db_eng directly
            df.to_sql(table_name, main_db_eng, if_exists="replace", index=False)
        del df

        # Start topic modeling process
        print("Starting preprocessing...")

        # Get data from database
        # Read directly using the main_db_eng
        df = pd.read_sql_table(table_name, main_db_eng)

        if LANGUAGE == "TR":
            # temizle
            tdm, sozluk, sayisal_veri, tokenizer = process_turkish_file(df, desired_columns, tokenizer,tokenizer_type=tokenizer_type)

        elif LANGUAGE == "EN":
            tdm, sozluk, sayisal_veri = process_english_file(df, desired_columns, LEMMATIZE)

        else:
            raise ValueError(f"Invalid language: {LANGUAGE}")

        # nmf
        W, H = run_nmf(
            num_of_topics=int(desired_topic_count),
            sparse_matrix=tdm,
            norm_thresh=0.005,
            nmf_method=nmf_type
        )

        word_pairs_out = False
        if word_pairs_out:
            # Calculate word co-occurrence matrix and save to output dir
            top_pairs = calc_word_cooccurrence(H, sozluk, base_dir, table_name, top_n=100, min_score=1, language=LANGUAGE, tokenizer=tokenizer)

        # generate topic distribution plot
        if gen_topic_distribution:
            gen_topic_dist(W, output_dir, table_name)

        # Find dominant words for each topic and dominant documents for each topic
        print("Generating topic groups...")
        if LANGUAGE == "TR":
            result = konu_analizi(
                H=H, 
                W=W, 
                konu_sayisi=int(desired_topic_count), 
                tokenizer=tokenizer, 
                documents=df[desired_columns], 
                topics_db_eng=topics_db_eng, 
                data_frame_name=table_name,
                word_per_topic=N_TOPICS,
                include_documents=True
            )
        elif LANGUAGE == "EN":
            result = konu_analizi(
                H=H,
                W=W,
                konu_sayisi=int(desired_topic_count),
                sozluk=sozluk,
                documents=df[desired_columns],
                topics_db_eng=topics_db_eng,
                data_frame_name=table_name,
                word_per_topic=N_TOPICS,
                include_documents=True
            )
        else:
            raise ValueError(f"Invalid language: {LANGUAGE}")

        # save result to json
        # Convert the topics_data format to the desired format
        topic_word_scores = save_doc_score_pair(base_dir,
                                                output_dir,
                                                table_name,
                                                result, H)

        # Calculate and save coherence scores

        coherence_scores = calculate_coherence_scores(topic_word_scores, output_dir, table_name)

        gen_cloud = True
        if gen_cloud:
            generate_wordclouds(result, output_dir, table_name)

        save_excel = True
        if save_excel:
            export_topics_to_excel(topic_word_scores, output_dir, table_name)

        print("Topic modeling completed successfully!")
        return {
            "state": "SUCCESS",
            "message": "Topic modeling completed successfully",
            "data_name": table_name,
            "topic_word_scores": topic_word_scores,
            # "topic_document_counts": {f"Topic {i+1}": count for i, count in enumerate(topic_counts)},
            # "plot_path": plot_path
            # "coherence_scores": coherence_scores
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Update queue status on error
        return {"state": "FAILURE", "message": str(e), "tablo_adi": table_name}


def run_standalone_nmf(
        filepath, table_name, desired_columns, desired_topic_count, LEMMATIZE, N_TOPICS, tokenizer_type, LANGUAGE, nmf_type,separator=","
):
    """
    Run the standalone NMF process with the given parameters.

    Args:
        filepath (str): Path to the input CSV file
        table_name (str): Name of the table to store the data
        desired_columns (str): Column name to analyze
        desired_topic_count (int): Number of topics to extract
        tokenizer_type (str): Type of tokenizer to use (wordpiece or bpe)
        nmf_type (str): NMF method to use ("opnmf" or "nmf")
    Returns:
        dict: Result of the process containing state, message, and topic word scores dictionary
    """
    start_time = time.time()
    print("Starting standalone NMF process...")
    # Initialize tokenizer once before processing
    tokenizer = init_tokenizer(tokenizer_type=tokenizer_type)

    result = process_file(
        filepath, table_name, desired_columns, desired_topic_count, LEMMATIZE, N_TOPICS,
        tokenizer=tokenizer,
        LANGUAGE=LANGUAGE,
        gen_topic_distribution=True,
        nmf_type=nmf_type,
        separator = separator
    )

    end_time = time.time()
    print(f"NMF process completed in {end_time - start_time:.2f} seconds")
    return result


if __name__ == "__main__":
    LEMMATIZE = True
    N_WORDS = 15
    DESIRED_TOPIC_COUNT = 12
    tokenizer_type = "bpe"  # "wordpiece" or "bpe"
    nmf_type = "nmf"
    LANGUAGE = "TR"
    separator = ";"
    filepath = "veri_setleri/playstore.csv"
    table_name = "PLAYSTORE" + f"_{nmf_type}_"+ tokenizer_type +"_"+str(DESIRED_TOPIC_COUNT)
    desired_columns = "REVIEW_TEXT"

    # Single run example
    run_standalone_nmf(
        filepath, table_name, desired_columns, DESIRED_TOPIC_COUNT, LEMMATIZE, N_WORDS, tokenizer_type, LANGUAGE, nmf_type,separator
    )
