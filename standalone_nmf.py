import os
import time

import pandas as pd
from sqlalchemy import create_engine

from functions.english.sozluk import sozluk_yarat
from functions.nmf import run_nmf
from functions.tfidf import tf_idf_generator, tfidf_hesapla
from functions.turkish.emoji_map import EmojiMap
from functions.turkish.konuAnalizi import konu_analizi
from functions.turkish.sayisallastir import veri_sayisallastir
from functions.turkish.temizle import metin_temizle
from functions.turkish.token_yarat import init_tokenizer, train_tokenizer
from utils.coherence_score import calculate_coherence_scores
from utils.export_excel import export_topics_to_excel
from utils.gen_cloud import generate_wordclouds
from utils.save_doc_score_pair import save_doc_score_pair
from utils.topic_dist import gen_topic_dist
from utils.word_cooccurrence import calc_word_cooccurrence


# rom utils.hierarchy_nmf import hierarchy_nmf


def process_turkish_file(df, desired_columns: str, tokenizer=None, tokenizer_type=None, emoji_map=None):
    """
    Process Turkish text data for topic modeling using NMF.

    This function performs text preprocessing, tokenization, and TF-IDF transformation
    specifically for Turkish language texts. It handles text cleaning, emoji mapping,
    tokenizer training, and vectorization.

    Args:
        df (pd.DataFrame): Input DataFrame containing Turkish text data
        desired_columns (str): Name of the column containing text to analyze
        tokenizer (optional): Pre-trained tokenizer instance. If None, a new tokenizer
                             will be initialized based on tokenizer_type
        tokenizer_type (str, optional): Type of tokenizer to use. Options: "bpe" or "wordpiece"
        emoji_map (EmojiMap, optional): Emoji mapping instance for emoji processing

    Returns:
        tuple: A tuple containing:
            - tdm (scipy.sparse matrix): Term-document matrix (TF-IDF transformed)
            - sozluk (list): Vocabulary list from the tokenizer
            - sayisal_veri (scipy.sparse matrix): Numerical representation of documents
            - tokenizer: Trained tokenizer instance
            - metin_array (list): Cleaned text array
            - emoji_map (EmojiMap): Emoji mapping instance used

    Raises:
        ValueError: If tokenizer_type is not supported
        KeyError: If desired_columns is not found in the DataFrame
    """

    metin_array = metin_temizle(df, desired_columns, emoji_map=emoji_map)
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

    return tdm, sozluk, sayisal_veri, tokenizer, metin_array, emoji_map


def process_english_file(df, desired_columns: str, lemmatize: bool):
    """
    Process English text data for topic modeling using NMF.

    This function performs text preprocessing and TF-IDF transformation specifically
    for English language texts. It creates a vocabulary dictionary and transforms
    the text data into numerical format suitable for topic modeling.

    Args:
        df (pd.DataFrame): Input DataFrame containing English text data
        desired_columns (str): Name of the column containing text to analyze
        lemmatize (bool): Whether to apply lemmatization during preprocessing.
                         If True, words are reduced to their base forms

    Returns:
        tuple: A tuple containing:
            - tdm (scipy.sparse matrix): Term-document matrix (TF-IDF transformed)
            - sozluk (dict): Vocabulary dictionary mapping words to indices
            - sayisal_veri (scipy.sparse matrix): TF-IDF transformed numerical data

    Raises:
        KeyError: If desired_columns is not found in the DataFrame
        ValueError: If the DataFrame is empty or contains no valid text data
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
    Process a file and perform comprehensive topic modeling analysis.

    This is the main processing function that handles file reading, data preprocessing,
    topic modeling using Non-negative Matrix Factorization (NMF), and result generation.
    It supports both Turkish and English languages with various output options.

    Args:
        filepath (str): Absolute path to the input file (CSV or Excel format)
        table_name (str): Unique identifier for the dataset, used for database storage
                         and output file naming
        desired_columns (str): Name of the column containing text data to analyze
        options (dict): Configuration dictionary with the following structure:
            {
                "LANGUAGE": str,                    # "TR" for Turkish, "EN" for English
                "DESIRED_TOPIC_COUNT": int,         # Number of topics to extract
                "N_TOPICS": int,                    # Top words per topic to display
                "LEMMATIZE": bool,                  # Lemmatize English text (ignored for Turkish)
                "tokenizer_type": str,              # "bpe" or "wordpiece" for Turkish
                "tokenizer": object,                # Pre-initialized tokenizer (optional)
                "nmf_type": str,                    # "opnmf" or "nmf" algorithm variant
                "gen_cloud": bool,                  # Generate word clouds for topics
                "save_excel": bool,                 # Export results to Excel format
                "gen_topic_distribution": bool,     # Generate topic distribution plots
                "filter_app": bool,                 # Filter data by application name
                "filter_app_name": str,             # App name to filter (if filter_app=True)
                "emoji_map": EmojiMap              # Emoji processing for Turkish texts
            }

    Returns:
        dict: Processing result containing:
            - state (str): "SUCCESS" if completed successfully, "FAILURE" if error occurred
            - message (str): Descriptive message about the processing outcome
            - data_name (str): Name of the processed dataset
            - topic_word_scores (dict): Dictionary mapping topic IDs to word-score pairs

    Raises:
        ValueError: If invalid language code or unsupported file format is provided
        FileNotFoundError: If the input file path does not exist
        KeyError: If required columns are missing from the input data
        Exception: For various processing errors (database, NMF computation, etc.)

    Note:
        - Creates SQLite databases in the 'instance' directory for data storage
        - Generates output files in the 'Output' directory
        - Supports CSV files with automatic delimiter detection and Excel files
        - Filters data for Turkish country code ('TR') when processing CSV files
        - Automatically handles file preprocessing (duplicate removal, null value handling)
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
            tdm, sozluk, sayisal_veri, tokenizer, metin_array, emoji_map = process_turkish_file(df,
                                                                                                desired_columns,
                                                                                                options["tokenizer"],
                                                                                                tokenizer_type=options["tokenizer_type"],
                                                                                                emoji_map=options["emoji_map"])

        elif options["LANGUAGE"] == "EN":
            tdm, sozluk, sayisal_veri = process_english_file(df, desired_columns, options["LEMMATIZE"])

        else:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}")

        # Create table-specific output directory to save everything under one folder
        table_output_dir = os.path.join(output_dir, table_name)
        os.makedirs(table_output_dir, exist_ok=True)

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
            top_pairs = calc_word_cooccurrence(H, sozluk, output_dir, table_name, top_n=100, min_score=1,
                                               language=options["LANGUAGE"], tokenizer=tokenizer)

        # Find dominant words for each topic and dominant documents for each topic
        print("Generating topic groups...")
        if options["LANGUAGE"] == "TR":
            result = konu_analizi(
                H=H,
                W=W,
                konu_sayisi=int(options["DESIRED_TOPIC_COUNT"]),
                sozluk=sozluk,
                tokenizer=tokenizer,
                documents=metin_array,
                topics_db_eng=topics_db_eng,
                data_frame_name=table_name,
                word_per_topic=options["N_TOPICS"],
                include_documents=True,
                emoji_map=emoji_map
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
        topic_word_scores = save_doc_score_pair(None,
                                                table_output_dir,
                                                table_name,
                                                result, H)

        # Calculate and save coherence scores
        coherence_scores = calculate_coherence_scores(topic_word_scores,
                                                      output_dir=table_output_dir,
                                                      column_name=desired_columns,
                                                      cleaned_data=metin_array,
                                                      table_name=table_name)

        # generate topic distribution plot
        if options["gen_topic_distribution"]:
            gen_topic_dist(W, table_output_dir, table_name)

        if options["gen_cloud"]:
            generate_wordclouds(result, table_output_dir, table_name)

        if options["save_excel"]:
            export_topics_to_excel(topic_word_scores, table_output_dir, table_name)

        '''new_hierarchy = hierarchy_nmf(W, tdm, selected_topic=1, desired_topic_count=options["DESIRED_TOPIC_COUNT"],
                                      nmf_method=options["nmf_type"], sozluk=sozluk, tokenizer=tokenizer,
                                      metin_array=metin_array, topics_db_eng=topics_db_eng, table_name=table_name,
                                      emoji_map=emoji_map, base_dir=base_dir, output_dir=output_dir)'''

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
    Execute the complete standalone NMF topic modeling pipeline.

    This is the main entry point for running topic modeling analysis without
    external dependencies like Celery or Redis. It initializes the tokenizer,
    measures execution time, and orchestrates the entire processing workflow.

    Args:
        filepath (str): Absolute path to the input data file (CSV or Excel)
        table_name (str): Unique identifier for the dataset used in database
                         storage and output file naming
        desired_columns (str): Name of the column containing text data to analyze
        options (dict): Comprehensive configuration dictionary with all processing
                       parameters. See process_file() documentation for detailed
                       options specification.

    Returns:
        dict: Complete processing result from process_file() containing:
            - state (str): Processing status ("SUCCESS" or "FAILURE")
            - message (str): Detailed status message
            - data_name (str): Processed dataset identifier
            - topic_word_scores (dict): Topic modeling results with word scores

    Side Effects:
        - Prints execution time and progress messages to console
        - Initializes tokenizer and adds it to the options dictionary
        - Creates output directories and database files as needed

    Example:
        >>> options = {
        ...     "LANGUAGE": "TR",
        ...     "DESIRED_TOPIC_COUNT": 5,
        ...     "tokenizer_type": "bpe",
        ...     "gen_cloud": True
        ... }
        >>> result = run_standalone_nmf("data.csv", "analysis", "text_column", options)
        >>> print(f"Status: {result['state']}")
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
    DESIRED_TOPIC_COUNT = 5
    tokenizer_type = "bpe"  # "wordpiece" or "bpe"
    nmf_type = "nmf"
    LANGUAGE = "TR"
    separator = ";"
    filepath = "veri_setleri/APPSTORE_APP_REVIEWSyeni_yeni.csv"
    filter_app_name = ""
    table_name = "APPSTORE" + f"_{nmf_type}_" + tokenizer_type + "_" + str(DESIRED_TOPIC_COUNT)
    desired_columns = "REVIEW"

    emj_map = EmojiMap()
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
        "filter_app": False,
        "filter_app_name": filter_app_name,
        "emoji_map": emj_map
    }

    run_standalone_nmf(filepath, table_name, desired_columns, options)
