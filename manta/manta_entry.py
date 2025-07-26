import os
import time
from pathlib import Path
import csv

import pandas as pd
from sqlalchemy import create_engine

from ._functions.common_language.emoji_processor import EmojiMap
from ._functions.common_language.topic_analyzer import konu_analizi

from ._functions.english.english_entry import process_english_file, process_english_file_from_generator
from ._functions.nmf import run_nmf
from ._functions.turkish.turkish_entry import process_turkish_file, process_turkish_file_from_generator
from ._functions.turkish.turkish_tokenizer_factory import init_tokenizer

from .utils.coherence_score import calculate_coherence_scores
from .utils.save_doc_score_pair import save_doc_score_pair
from .utils.save_word_score_pair import save_word_score_pair
from .utils.visualizer import create_visualization
from .utils.json_to_excel import convert_json_to_excel


def read_csv_generator(filepath, options):
    """
    Generator function to read CSV file line by line with filtering.
    
    Args:
        filepath (str): Path to CSV file
        options (dict): Configuration options containing separator and filter settings
        
    Yields:
        dict: Row data as dictionary
    """
    # Handle CSV preprocessing if needed
    actual_filepath = filepath
    if hasattr(options, 'preprocess_csv') and options.get('preprocess_csv', False):
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.read()
            # replace "|" with ";"
            data = data.replace("|", ";")
            # remove tab and null characters
            data = data.replace("\t", "")
            data = data.replace("\x00", "")
        
        new_filepath = filepath.replace(".csv", "_new.csv")
        with open(new_filepath, "w", encoding="utf-8") as f_out:
            f_out.write(data)
        actual_filepath = new_filepath
    
    # Read CSV file line by line
    try:
        with open(actual_filepath, 'r', encoding='utf-8') as csvfile:
            # Use csv.Sniffer to detect delimiter if needed, otherwise use provided separator
            separator = options.get("separator", ",")
            
            reader = csv.DictReader(csvfile, delimiter=separator)
            
            for row in reader:
                # Apply filters if specified
                try:
                    if options.get("filter_app", False):
                        filter_options = options.get("data_filter_options", {})
                        
                        # Country filter
                        if filter_options.get("filter_app_country", "") != "":
                            country_col = filter_options.get("filter_app_country_column", "")
                            expected_country = filter_options.get("filter_app_country", "")
                            if country_col in row:
                                if row[country_col].upper() != expected_country:
                                    continue
                        
                        # App name filter
                        if filter_options.get("filter_app_name", "") != "":
                            app_col = filter_options.get("filter_app_column", "")
                            expected_app = filter_options.get("filter_app_name", "")
                            if app_col in row:
                                if row[app_col] != expected_app:
                                    continue
                except KeyError:
                    # Skip row if filter columns are missing
                    continue
                
                # Yield the filtered row
                yield row
                
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

def process_file(
    filepath: str,
    table_name: str,
    desired_columns: str,
    options: dict,
    output_base_dir: str = None,
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
    if output_base_dir is None:
        # Use current working directory instead of package directory
        base_dir = Path.cwd()
    else:
        base_dir = Path(output_base_dir).resolve()

    program_output_dir = base_dir / "TopicAnalysis"
    instance_path = program_output_dir / "instance"
    output_dir = program_output_dir / "Output"

    # Create necessary directories first
    instance_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Database configurations using standard SQLAlchemy
    topics_db_eng = create_engine(
        f'sqlite:///{instance_path / "topics.db"}'
    )
    main_db_eng = create_engine(
        f'sqlite:///{instance_path / "scopus.db"}'
    )  # Main data DB

    try:
        print(f"Starting topic modeling for {table_name}")

        # Clean up the desired_columns
        desired_columns = desired_columns.strip() if desired_columns else None

        # Read the input file using generators for memory efficiency
        print("Reading input file using generator...")
        
        use_generators = False  # Flag to enable generator-based processing
        
        if use_generators and str(filepath).endswith(".csv"):
            # Use generator-based CSV reading
            csv_gen = read_csv_generator(filepath, options)
            row_count = 0
            for _ in csv_gen:
                row_count += 1
            print(f"File has {row_count} rows.")
            # Recreate generator since we consumed it for counting
            csv_gen = read_csv_generator(filepath, options)
            df = None  # We don't use DataFrame in generator mode
        else:
            # Fallback to original DataFrame-based processing
            if str(filepath).endswith(".csv"):
                preprocess_csv = False
                if preprocess_csv:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = f.read()
                        # replace "|" with ";"
                        data = data.replace("|", ";")
                        # remove tab and null characters
                        data = data.replace("\t", "")
                        data = data.replace("\x00", "")
                        # save the modified data back to the new file
                    new_filepath = filepath.replace(".csv", "_new.csv")
                    with open(new_filepath, "w", encoding="utf-8") as f_out:
                        f_out.write(data)
                    filepath = new_filepath
                # Read the CSV file with the specified separator

                df = pd.read_csv(
                    filepath,
                    encoding="utf-8",
                    sep=options["separator"],
                    engine="python",
                    on_bad_lines="skip",
                )
                # get rows where it is country is TR
                try:
                    if options["filter_app"]:
                        if options["data_filter_options"]["filter_app_country"] != "":
                            df = df[df[options["data_filter_options"]["filter_app_country_column"]].str.upper()
                                    == options["data_filter_options"]["filter_app_country"]]
                        if options["data_filter_options"]["filter_app_name"] != "":
                            df = df[df[options["data_filter_options"]["filter_app_column"]] == options["data_filter_options"][
                                "filter_app_name"]]
                except:
                    print("Either no filter is applied or the filter columns are not present in the data.")

            else:
                df = pd.read_excel(filepath)

            # Add to main database
            # INSTEAD OF SAVING WHOLE TABLE TO DATABASE, SAVE ONLY THE DESIRED COLUMNS
            df = df[desired_columns]
            # Use double brackets to select columns
            df = df.drop_duplicates()
            df = df.dropna()
            print(f"File has {len(df)} rows.")

            options["save_to_db"] = False
            if options["save_to_db"]:
                print("Adding data to main database...")
                # Check if table exists using a SQL query instead of direct table read
                tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
                existing_tables = pd.read_sql_query(tables_query, main_db_eng)
                if table_name in existing_tables["name"].tolist():
                    # Use the main_db_eng directly
                    df.to_sql(table_name, main_db_eng, if_exists="replace", index=False)
                elif table_name not in existing_tables["name"].tolist():
                    df.to_sql(table_name, main_db_eng, if_exists="replace", index=False)
                # Get data from database
                # Read directly using the main_db_eng
                del df
                df = pd.read_sql_table(table_name, main_db_eng)
            else:
                df = pd.DataFrame(df)
                print("Not saving data to main database...")
            csv_gen = None  # Not using generator in DataFrame mode

        # Start topic modeling process
        print("Starting preprocessing...")

        if use_generators and csv_gen is not None:
            # Use generator-based processing
            if options["LANGUAGE"] == "TR":
                tdm, sozluk, sayisal_veri, options["tokenizer"], metin_array, emoji_map = (
                    process_turkish_file_from_generator(
                        csv_gen,
                        desired_columns,
                        options["tokenizer"],
                        tokenizer_type=options["tokenizer_type"],
                        emoji_map=options["emoji_map"],
                    )
                )

            elif options["LANGUAGE"] == "EN":
                tdm, sozluk, sayisal_veri, tokenizer, metin_array, emoji_map = process_english_file_from_generator(
                    csv_gen,
                    desired_columns,
                    options["LEMMATIZE"],
                    emoji_map=options["emoji_map"],
                )

            else:
                raise ValueError(f"Invalid language: {options['LANGUAGE']}")
        else:
            # Use traditional DataFrame-based processing
            if options["LANGUAGE"] == "TR":
                tdm, sozluk, sayisal_veri, options["tokenizer"], metin_array, emoji_map = (
                    process_turkish_file(
                        df,
                        desired_columns,
                        options["tokenizer"],
                        tokenizer_type=options["tokenizer_type"],
                        emoji_map=options["emoji_map"],
                    )
                )

            elif options["LANGUAGE"] == "EN":
                tdm, sozluk, sayisal_veri, tokenizer, metin_array, emoji_map = process_english_file(
                    df,
                    desired_columns,
                    options["LEMMATIZE"],
                    tokenizer=options.get("tokenizer"),
                    tokenizer_type=options.get("tokenizer_type", "wordpiece"),
                    emoji_map=options["emoji_map"],
                )

            else:
                raise ValueError(f"Invalid language: {options['LANGUAGE']}")

        # Create table-specific output directory to save everything under one folder
        table_output_dir = output_dir / table_name
        table_output_dir.mkdir(parents=True, exist_ok=True)

        # nmf
        W, H = run_nmf(
            num_of_topics=int(options["DESIRED_TOPIC_COUNT"]),
            sparse_matrix=tdm,
            norm_thresh=0.005,
            nmf_method=options["nmf_type"],
        )

        # Find dominant words for each topic and dominant documents for each topic
        print("Generating topic groups...")
        if options["LANGUAGE"] == "TR":
            word_result, document_result = konu_analizi(
                H=H,
                W=W,
                konu_sayisi=int(options["DESIRED_TOPIC_COUNT"]),
                sozluk=sozluk,
                tokenizer=options["tokenizer"],
                documents=metin_array,
                topics_db_eng=topics_db_eng,
                data_frame_name=table_name,
                word_per_topic=options["N_TOPICS"],
                include_documents=True,
                emoji_map=options["emoji_map"],
                output_dir=table_output_dir,
            )
        elif options["LANGUAGE"] == "EN":
            word_result, document_result = konu_analizi(
                H=H,
                W=W,
                konu_sayisi=int(options["DESIRED_TOPIC_COUNT"]),
                sozluk=sozluk,
                documents=metin_array,
                topics_db_eng=topics_db_eng,
                data_frame_name=table_name,
                word_per_topic=options["N_TOPICS"],
                include_documents=True,
                emoji_map=options["emoji_map"],
                output_dir=table_output_dir,
            )
        else:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}")

        # save result to json
        # Convert the topics_data format to the desired format
        topic_word_scores = save_word_score_pair(
            base_dir=None,
            output_dir=table_output_dir,
            table_name=table_name,
            topics_data=word_result,
            result=None,
            data_frame_name=table_name,
            topics_db_eng=topics_db_eng,
        )
        # save document result to json
        topic_doc_scores = save_doc_score_pair(
            document_result,
            base_dir=None,
            output_dir=table_output_dir,
            table_name=table_name,
            data_frame_name=table_name,
        )

        # Calculate and save coherence scores
        coherence_scores = calculate_coherence_scores(
            topic_word_scores,
            output_dir=table_output_dir,
            column_name=desired_columns,
            cleaned_data=metin_array,
            table_name=table_name,
        )

        visual_returns = create_visualization(
            W,
            H,
            sozluk,
            table_output_dir,
            table_name,
            options,
            word_result,
            topic_word_scores,
            metin_array,
            topics_db_eng,
            options["emoji_map"],
            program_output_dir,
            output_dir,
        )

        save_to_excel = True
        if save_to_excel:
            # save jsons to excel format
            convert_json_to_excel(
                word_json_data=topic_word_scores,
                doc_json_data=topic_doc_scores,
                output_dir=table_output_dir,
                data_frame_name=table_name,
                total_docs_count=len(metin_array),
            )


        print("Topic modeling completed successfully!")

        return {
            "state": "SUCCESS",
            "message": "Topic modeling completed successfully",
            "data_name": table_name,
            "topic_word_scores": topic_word_scores,
            "topic_doc_scores": topic_doc_scores,
            "coherence_scores": coherence_scores,
            "topic_dist_img": visual_returns[0] if options["gen_topic_distribution"] else None,
            "topic_document_counts": visual_returns[1] if options["gen_topic_distribution"] else None
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Update queue status on error
        return {"state": "FAILURE", "message": str(e), "tablo_adi": table_name}


def run_standalone_nmf(
    filepath, table_name, desired_columns, options, output_base_dir=None
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

    """
    start_time = time.time()
    print("Starting standalone NMF process...")
    # Initialize tokenizer once before processing
    tokenizer = init_tokenizer(tokenizer_type=options["tokenizer_type"])
    options["tokenizer"] = tokenizer
    
    # Convert boolean emoji_map to EmojiMap object if needed
    if options.get("emoji_map") is True:
        options["emoji_map"] = EmojiMap()
    elif options.get("emoji_map") is False:
        options["emoji_map"] = None

    result = process_file(
        filepath, table_name, desired_columns, options, output_base_dir
    )

    end_time = time.time()
    print(f"Topic analysis process completed in {end_time - start_time:.2f} seconds")
    return result


if __name__ == "__main__":
    LEMMATIZE = True
    N_WORDS = 15
    DESIRED_TOPIC_COUNT = 5
    tokenizer_type = "bpe"  # "wordpiece" or "bpe"
    nmf_type = "nmf"
    filepath = "veri_setleri/APPSTORE_APP_REVIEWSyeni_yeni.csv"
    data_name = filepath.split("/")[-1].split(".")[0].split("_")[0]
    LANGUAGE = "TR"
    separator = "|"
    filter_app_name = ""
    table_name = (
        data_name + f"_{nmf_type}_" + tokenizer_type + "_" + str(DESIRED_TOPIC_COUNT)
    )
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
        "word_pairs_out": True,
        "gen_topic_distribution": True,
        "emoji_map": True,
        "filter_app" : False,
        "data_filter_options": {
            "filter_app_name": "",
            "filter_app_column": "PACKAGE_NAME",
            "filter_app_country": "TR",
            "filter_app_country_column": "COUNTRY",
        }
    }
    '''  
        # Co-occurrence analysis options
        "cooccurrence_method": "sliding_window",  # "nmf" or "sliding_window"
        "cooccurrence_window_size": 10,           # Sliding window size for co-occurrence
        "cooccurrence_min_count": 2,             # Minimum word frequency threshold
        "cooccurrence_max_vocab": None,          # Maximum vocabulary size (None = no limit)
        "cooccurrence_heatmap_size": 20,         # Number of words in heatmap
        "cooccurrence_top_n": 100,               # Number of top pairs to save
        "cooccurrence_batch_size": 1000,         # Batch size for memory efficiency
        "cooccurrence_min_score": 1,             # Minimum score for NMF method
    }
    '''
    run_standalone_nmf(filepath, table_name, desired_columns, options)
