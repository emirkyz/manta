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


def process_turkish_file(df, desired_columns: str, tokenizer=None, tokenizer_type=None):
    """
    Process Turkish text data through the complete preprocessing pipeline.
    
    Args:
        df (pandas.DataFrame): Input dataframe containing Turkish text
        desired_columns (str): Column name containing text to analyze
        tokenizer (optional): Pre-trained tokenizer object
        tokenizer_type (str): Either "bpe" or "wordpiece"
    
    Returns:
        tuple: (tdm, sozluk, sayisal_veri, tokenizer)
            - tdm: TF-IDF document-term matrix
            - sozluk: Vocabulary list
            - sayisal_veri: Numerical representation of documents
            - tokenizer: Trained tokenizer object
    """
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


def process_english_file(df, desired_columns: str, lemmatize: bool):
    """
    Process English text data with lemmatization and dictionary-based approaches.
    
    Args:
        df (pandas.DataFrame): Input dataframe containing English text
        desired_columns (str): Column name containing text to analyze
        lemmatize (bool): Whether to apply lemmatization
    
    Returns:
        tuple: (tdm, sozluk, sayisal_veri)
            - tdm: TF-IDF document-term matrix (same as sayisal_veri)
            - sozluk: Dictionary/vocabulary of terms
            - sayisal_veri: TF-IDF weighted document-term matrix
    """
    sozluk, N = sozluk_yarat(df, desired_columns, lemmatize=lemmatize)
    sayisal_veri = tfidf_hesapla(N, sozluk=sozluk, data=df, alanadi=desired_columns, output_dir=None,
                                 lemmatize=lemmatize)
    tdm = sayisal_veri

    return tdm, sozluk, sayisal_veri


def process_file(
        filepath: str,
        table_name: str,
        desired_columns: str,
        options: dict
) -> dict:
    """
    Complete topic modeling pipeline from file input to results.
    
    Args:
        filepath (str): Path to input CSV/Excel file
        table_name (str): Unique identifier for this analysis run
        desired_columns (str): Column name containing text data
        options (dict): Configuration dictionary containing:
            - LANGUAGE (str): "TR" for Turkish, "EN" for English
            - DESIRED_TOPIC_COUNT (int): Number of topics to extract
            - N_TOPICS (int): Number of top words per topic to display
            - tokenizer_type (str): "bpe" or "wordpiece" for Turkish
            - nmf_type (str): "nmf" or "opnmf" algorithm choice
            - LEMMATIZE (bool): Enable lemmatization (mainly for English)
            - gen_topic_distribution (bool): Generate topic distribution plots
            - gen_cloud (bool): Generate word clouds
            - save_excel (bool): Export results to Excel
    
    Returns:
        dict: Results containing:
            - state: "SUCCESS" or "FAILURE"
            - message: Status message
            - data_name: Analysis identifier
            - topic_word_scores: Topic-word associations
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
            with open(filepath, 'r', encoding='utf-8') as f:
                data = f.read()
                # replace "|" with ";"
                data = data.replace("|", ";")
                # remove tab and null characters
                data = data.replace("\t", "")
                data = data.replace("\x00", "")
                # save the modified data back to the new file
            new_filepath = filepath.replace(".csv", "_new.csv")
            with open(new_filepath, 'w', encoding='utf-8') as f_out:
                f_out.write(data)
            filepath = new_filepath
            # Read the CSV file with the specified separator

            df = pd.read_csv(filepath, encoding="utf-8", sep=None, engine="python", on_bad_lines="skip")
            # get rows where it is country is TR
            df = df[df['COUNTRY'] == 'TR']
            if options["filter_app"]:
                df = df[df['APP_NAME_ABBR'] == options["filter_app_name"]]

        else:
            df = pd.read_excel(filepath)

        # Add to main database

        # INSTEAD OF SAVING WHOLE TABLE TO DATABASE, SAVE ONLY THE DESIRED COLUMNS
        # app_col = "PACKAGE_NAME"
        # get only bip
        # df = df[df[app_col] == "com.turkcell.bip"]
        # drop duplicates based on ID column

        # df = df.drop_duplicates(subset=['ID'])
        df = df[desired_columns]
        # Use double brackets to select columns
        df = df.drop_duplicates()
        df = df.dropna()

        '''
        # remove duplicates
        count_of_duplicates = df.duplicated().sum()
        total_rows = len(df)
        if total_rows*0.9 < count_of_duplicates:
            print(f"Warning: {count_of_duplicates} duplicates found in the data, which is more than 90% of the total rows ({total_rows}).")
            df = df.drop_duplicates()
        '''
        # df = df.drop_duplicates()
        print(f"File has {len(df)} rows.")

        print("Adding data to main database...")
        # Check if table exists using a SQL query instead of direct table read
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        existing_tables = pd.read_sql_query(tables_query, main_db_eng)
        if table_name in existing_tables['name'].tolist():
            # Use the main_db_eng directly
            df.to_sql(table_name, main_db_eng, if_exists="replace", index=False)
        elif table_name not in existing_tables['name'].tolist():
            df.to_sql(table_name, main_db_eng, if_exists="replace", index=False)
        del df

        # Start topic modeling process
        print("Starting preprocessing...")

        # Get data from database
        # Read directly using the main_db_eng
        df = pd.read_sql_table(table_name, main_db_eng)

        if options["LANGUAGE"] == "TR":
            # temizle
            tdm, sozluk, sayisal_veri, tokenizer = process_turkish_file(df, desired_columns, options["tokenizer"],
                                                                        tokenizer_type=options["tokenizer_type"])

        elif options["LANGUAGE"] == "EN":
            tdm, sozluk, sayisal_veri = process_english_file(df, desired_columns, options["LEMMATIZE"])

        else:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}")

        # nmf
        W, H = run_nmf(
            num_of_topics=int(options["DESIRED_TOPIC_COUNT"]),
            sparse_matrix=tdm,
            norm_thresh=0.005,
            nmf_method=options["nmf_type"]
        )

        word_pairs_out = False
        if word_pairs_out:
            # Calculate word co-occurrence matrix and save to output dir
            top_pairs = calc_word_cooccurrence(H, sozluk, base_dir, table_name, top_n=100, min_score=1,
                                               language=options["LANGUAGE"], tokenizer=tokenizer)

        # generate topic distribution plot
        if options["gen_topic_distribution"]:
            gen_topic_dist(W, output_dir, table_name)

        # Find dominant words for each topic and dominant documents for each topic
        print("Generating topic groups...")
        if options["LANGUAGE"] == "TR":
            result = konu_analizi(
                H=H,
                W=W,
                konu_sayisi=int(options["DESIRED_TOPIC_COUNT"]),
                tokenizer=tokenizer,
                documents=df[desired_columns],
                topics_db_eng=topics_db_eng,
                data_frame_name=table_name,
                word_per_topic=options["N_TOPICS"],
                include_documents=True
            )
        elif options["LANGUAGE"] == "EN":
            result = konu_analizi(
                H=H,
                W=W,
                konu_sayisi=int(options["DESIRED_TOPIC_COUNT"]),
                sozluk=sozluk,
                documents=df[desired_columns],
                topics_db_eng=topics_db_eng,
                data_frame_name=table_name,
                word_per_topic=options["N_TOPICS"],
                include_documents=True
            )
        else:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}")

        # save result to json
        # Convert the topics_data format to the desired format
        topic_word_scores = save_doc_score_pair(base_dir,
                                                output_dir,
                                                table_name,
                                                result, H)

        # Calculate and save coherence scores

        coherence_scores = calculate_coherence_scores(topic_word_scores, output_dir, table_name)

        if options["gen_cloud"]:
            generate_wordclouds(result, output_dir, table_name)

        if options["save_excel"]:
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
        filepath, table_name, desired_columns, options
):
    """
    Simplified entry point for NMF topic modeling.
    
    Args:
        filepath (str): Path to input file
        table_name (str): Analysis identifier
        desired_columns (str): Text column name
        options (dict): Configuration dictionary containing:
            - LEMMATIZE (bool): Enable lemmatization
            - N_TOPICS (int): Words per topic
            - DESIRED_TOPIC_COUNT (int): Number of topics to extract
            - tokenizer_type (str): "bpe" or "wordpiece" for Turkish
            - nmf_type (str): "nmf" or "opnmf" algorithm choice
            - LANGUAGE (str): "TR" for Turkish, "EN" for English
            - separator (str): CSV separator character
            - gen_topic_distribution (bool): Generate distribution plots
    
    Returns:
        dict: Process results with timing information
    """
    start_time = time.time()
    print("Starting standalone NMF process...")
    # Initialize tokenizer once before processing
    tokenizer = init_tokenizer(tokenizer_type=options["tokenizer_type"])
    options["tokenizer"] = tokenizer

    result = process_file(
        filepath, table_name, desired_columns, options
    )

    end_time = time.time()
    print(f"NMF process completed in {end_time - start_time:.2f} seconds")
    return result


if __name__ == "__main__":
    LEMMATIZE = True
    N_WORDS = 15
    DESIRED_TOPIC_COUNT = 3
    tokenizer_type = "bpe"  # "wordpiece" or "bpe"
    nmf_type = "nmf"
    LANGUAGE = "TR"
    separator = ";"
    filepath = "veri_setleri/APPSTORE_APP_REVIEWSyeni_yeni.csv"
    filter_app_name = "BiP"
    table_name = "APPSTORE" + f"_{filter_app_name}" + f"_{nmf_type}_" + tokenizer_type + "_" + str(DESIRED_TOPIC_COUNT)
    desired_columns = "REVIEW"

    options = {
        "LEMMATIZE": LEMMATIZE,
        "N_TOPICS": N_WORDS,
        "DESIRED_TOPIC_COUNT": DESIRED_TOPIC_COUNT,
        "tokenizer_type": tokenizer_type,
        "tokenizer": None,
        "nmf_type": nmf_type,
        "LANGUAGE": LANGUAGE,
        "separator": separator,
        "gen_cloud": True,
        "save_excel": True,
        "word_pairs_out": False,
        "gen_topic_distribution": True,
        "filter_app": True,
        "filter_app_name": filter_app_name
    }

    # Single run example
    run_standalone_nmf(
        filepath, table_name, desired_columns, options
    )
