import os
import time
from typing import Dict, Any, Optional

import pandas as pd

from ._functions.common_language.emoji_processor import EmojiMap
from ._functions.common_language.topic_extractor import topic_extract

from ._functions.english.english_entry import process_english_file
from ._functions.nmf import run_nmf
from ._functions.turkish.turkish_entry import process_turkish_file
from ._functions.turkish.turkish_tokenizer_factory import init_tokenizer

from .utils.analysis.coherence_score import calculate_coherence_scores
from .utils.export.save_doc_score_pair import save_doc_score_pair
from .utils.export.save_word_score_pair import save_word_score_pair
from .utils.visualization.visualizer import create_visualization
from .utils.export.json_to_excel import convert_json_to_excel
from .utils.database.database_manager import DatabaseManager
from .utils.console.console_manager import ConsoleManager


def _validate_inputs(filepath: str, desired_columns: str, options: Dict[str, Any]) -> None:
    """
    Validate input parameters for processing.
    
    Args:
        filepath: Path to input file
        desired_columns: Column name containing text data
        options: Configuration options
        
    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    if not desired_columns or not desired_columns.strip():
        raise ValueError("desired_columns cannot be empty")
        
    required_options = ["LANGUAGE", "DESIRED_TOPIC_COUNT", "N_TOPICS"]
    for option in required_options:
        if option not in options:
            raise ValueError(f"Missing required option: {option}")
    
    if options["LANGUAGE"] not in ["TR", "EN"]:
        raise ValueError(f"Invalid language: {options['LANGUAGE']}. Must be 'TR' or 'EN'")


def _load_data_file(filepath: str, options: Dict[str, Any], console: Optional[ConsoleManager] = None) -> pd.DataFrame:
    """
    Load data from CSV or Excel file.
    
    Args:
        filepath: Path to input file
        options: Configuration options containing separator and filter settings
        console: Console manager for status messages
        
    Returns:
        Loaded DataFrame
    """
    if console:
        console.print_status("Reading input file...", "processing")
    else:
        print("Reading input file...")
    
    if str(filepath).endswith(".csv"):
        # Read the CSV file with the specified separator
        df = pd.read_csv(
            filepath,
            encoding="utf-8",
            sep=options["separator"],
            engine="python",
            on_bad_lines="skip",
        )

    elif str(filepath).endswith(".xlsx") or str(filepath).endswith(".xls"):
        df = pd.read_excel(filepath)

    # Apply data filters if specified
    try:
        if options.get("filter_app", False):
            filter_options = options.get("data_filter_options", {})
            if filter_options.get("filter_app_country", ""):
                country_col = filter_options.get("filter_app_country_column", "")
                if country_col in df.columns:
                    df = df[df[country_col].str.upper() == filter_options["filter_app_country"]]
                    if console:
                        console.print_status(f"Applied country filter: {filter_options['filter_app_country']}", "info")
                else:
                    msg = f"Warning: Filter column '{country_col}' not found in data"
                    if console:
                        console.print_status(msg, "warning")
                    else:
                        print(msg)

            if filter_options.get("filter_app_name", ""):
                app_col = filter_options.get("filter_app_column", "")
                if app_col in df.columns:
                    df = df[df[app_col] == filter_options["filter_app_name"]]
                    if console:
                        console.print_status(f"Applied app filter: {filter_options['filter_app_name']}", "info")
                else:
                    msg = f"Warning: Filter column '{app_col}' not found in data"
                    if console:
                        console.print_status(msg, "warning")
                    else:
                        print(msg)
    except KeyError as e:
        msg = f"Warning: Missing filter configuration: {e}"
        if console:
            console.print_status(msg, "warning")
        else:
            print(msg)
    except Exception as e:
        msg = f"Warning: Error applying data filters: {e}"
        if console:
            console.print_status(msg, "warning")
        else:
            print(msg)

    return df


def _preprocess_dataframe(df: pd.DataFrame, desired_columns: str, options: Dict[str, Any], main_db_eng, table_name: str, console: Optional[ConsoleManager] = None) -> pd.DataFrame:
    """
    Preprocess the loaded DataFrame.
    
    Args:
        df: Raw DataFrame
        desired_columns: Column containing text data
        options: Configuration options
        main_db_eng: Database engine for main data
        table_name: Name for database table
        console: Console manager for status messages
        
    Returns:
        Preprocessed DataFrame
    """
    if console:
        console.print_status("Preprocessing data...", "processing")
    
    # Select only desired columns and validate they exist
    if desired_columns not in df.columns:
        available_columns = ", ".join(df.columns.tolist())
        raise KeyError(f"Column '{desired_columns}' not found in data. Available columns: {available_columns}")
    
    df = df[desired_columns]
    
    # Remove duplicates and null values
    initial_count = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    
    if len(df) == 0:
        raise ValueError("No data remaining after removing duplicates and null values")
    
    if len(df) < initial_count * 0.1:
        msg = f"Warning: Only {len(df)} rows remain from original {initial_count} after preprocessing"
        if console:
            console.print_status(msg, "warning")
        else:
            print(msg)

    msg = f"Preprocessed dataset has {len(df)} rows"
    if console:
        console.print_status(msg, "info")
    else:
        print(f"File has {len(df)} rows.")

    # Handle database persistence
    df = DatabaseManager.handle_dataframe_persistence(
        df, table_name, main_db_eng, save_to_db=options["save_to_db"]
    )
    
    return df


def _perform_text_processing(df: pd.DataFrame, desired_columns: str, options: Dict[str, Any], console: Optional[ConsoleManager] = None):
    """
    Perform language-specific text processing and feature extraction.
    
    Args:
        df: Preprocessed DataFrame
        desired_columns: Column containing text data
        options: Configuration options
        console: Console manager for status messages
        
    Returns:
        Tuple of (tdm, vocab, counterized_data, text_array, updated_options)
    """
    if console:
        console.print_status(f"Starting text processing ({options['LANGUAGE']})...", "processing")
    else:
        print("Starting preprocessing...")
    
    if options["LANGUAGE"] == "TR":
        tdm, vocab, counterized_data, text_array, options["tokenizer"], options["emoji_map"] = (
            process_turkish_file(
                df,
                desired_columns,
                options["tokenizer"],
                tokenizer_type=options["tokenizer_type"],
                emoji_map=options["emoji_map"],
            )
        )
    elif options["LANGUAGE"] == "EN":
        tdm, vocab, counterized_data, text_array, options["emoji_map"] = process_english_file(
            df,
            desired_columns,
            options["LEMMATIZE"],
            emoji_map=options["emoji_map"],
        )
    else:
        raise ValueError(f"Invalid language: {options['LANGUAGE']}")
    
    if console:
        console.print_status("Text processing completed", "success")
    
    return tdm, vocab, counterized_data, text_array, options


def _perform_topic_modeling(tdm, options: Dict[str, Any], vocab, text_array, df: pd.DataFrame, desired_columns: str, db_config, table_name: str, table_output_dir, console: Optional[ConsoleManager] = None):
    """
    Perform NMF topic modeling and analysis.
    
    Args:
        console: Console manager for status messages
    
    Returns:
        Tuple of (topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result)
    """
    if console:
        console.print_status(f"Starting NMF processing ({options['nmf_type'].upper()})...", "processing")
    else:
        print("Starting NMF processing...")
    
    # nmf
    nmf_output = run_nmf(
        num_of_topics=int(options["DESIRED_TOPIC_COUNT"]),
        sparse_matrix=tdm,
        norm_thresh=0.005,
        nmf_method=options["nmf_type"],
    )

    if console:
        console.print_status("Extracting topics from NMF results...", "processing")
    else:
        print("Generating topic groups...")
        
    if options["LANGUAGE"] == "TR":
        word_result, document_result = topic_extract(
            H=nmf_output["H"],
            W=nmf_output["W"],
            doc_word_pairs=nmf_output.get("S", None),
            topic_count=int(options["DESIRED_TOPIC_COUNT"]),
            vocab=vocab,
            tokenizer=options["tokenizer"],
            documents=text_array,
            db_config=db_config,
            data_frame_name=table_name,
            word_per_topic=options["N_TOPICS"],
            include_documents=True,
            emoji_map=options["emoji_map"],
        )
    elif options["LANGUAGE"] == "EN":
        word_result, document_result = topic_extract(
            H=nmf_output["H"],
            W=nmf_output["W"],
            doc_word_pairs=nmf_output.get("S", None),
            topic_count=int(options["DESIRED_TOPIC_COUNT"]),
            vocab=vocab,
            documents=[str(doc).strip() for doc in df[desired_columns]],
            db_config=db_config,
            data_frame_name=table_name,
            word_per_topic=options["N_TOPICS"],
            include_documents=True,
            emoji_map=options["emoji_map"],
        )
    else:
        raise ValueError(f"Invalid language: {options['LANGUAGE']}")

    if console:
        console.print_status("Saving topic results...", "processing")
    else:
        print("Saving topic results...")
        
    # Convert the topics_data format to the desired format
    topic_word_scores = save_word_score_pair(
        base_dir=None,
        output_dir=table_output_dir,
        table_name=table_name,
        topics_data=word_result,
        result=None,
        data_frame_name=table_name,
        topics_db_eng=db_config.topics_db_engine,
    )
    # save document result to json
    topic_doc_scores = save_doc_score_pair(
        document_result,
        base_dir=None,
        output_dir=table_output_dir,
        table_name=table_name,
        data_frame_name=table_name,
    )

    if console:
        console.print_status("Calculating coherence scores...", "processing")
    else:
        print("Calculating coherence scores...")
        
    # Calculate and save coherence scores
    coherence_scores = calculate_coherence_scores(
        topic_word_scores,
        output_dir=table_output_dir,
        column_name=desired_columns,
        cleaned_data=text_array,
        table_name=table_name,
    )
    
    return topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result


def _generate_outputs(nmf_output, vocab, table_output_dir, table_name: str, options: Dict[str, Any], word_result, topic_word_scores, text_array, topics_db_eng, program_output_dir, output_dir, topic_doc_scores, console: Optional[ConsoleManager] = None):
    """
    Generate visualizations and output files.
    
    Args:
        console: Console manager for status messages
    
    Returns:
        Visual returns from visualization generation
    """
    if console:
        console.print_status("Generating visualizations and exports...", "processing")
    else:
        print("Generating visual outputs.")
        
    visual_returns = create_visualization(
        nmf_output["W"],
        nmf_output["H"],
        vocab,
        table_output_dir,
        table_name,
        options,
        word_result,
        topic_word_scores,
        text_array,
        topics_db_eng,
        options["emoji_map"],
        program_output_dir,
        output_dir,
    )

    save_to_excel = True
    if save_to_excel:
        if console:
            console.print_status("Exporting results to Excel...", "processing")
        # save jsons to excel format
        convert_json_to_excel(
            word_json_data=topic_word_scores,
            doc_json_data=topic_doc_scores,
            output_dir=table_output_dir,
            data_frame_name=table_name,
            total_docs_count=len(text_array),
        )
    
    if console:
        console.print_status("Output generation completed", "success")
    
    return visual_returns


def process_file(
    filepath: str,
    table_name: str,
    desired_columns: str,
    options: Dict[str, Any],
    output_base_dir: Optional[str] = None,
    console: Optional[ConsoleManager] = None,
) -> Dict[str, Any]:
    """
    Process a file and perform NMF topic modeling analysis.
    
    This function handles the complete topic modeling pipeline including data loading,
    preprocessing, NMF analysis, and output generation. It validates inputs, processes
    text according to language, performs topic modeling, and generates visualizations.
    
    Args:
        filepath: Path to input CSV or Excel file containing the text data for analysis
        table_name: Unique identifier used for naming output files and database tables
        desired_columns: Column name in the input file containing the text to analyze
        options: Dictionary containing processing configuration parameters including:
            - LANGUAGE: Text language ("TR" or "EN")
            - DESIRED_TOPIC_COUNT: Number of topics to extract 
            - N_TOPICS: Number of top words per topic
            - tokenizer_type: Type of tokenizer for Turkish text (bpe or wordpiece)
            - gen_cloud: Whether to generate word clouds
            - gen_topic_distribution: Whether to generate topic distribution plots
            - save_to_db: Whether to persist data to database
        output_base_dir: Base directory for outputs (optional). Defaults to current directory.
    
    Returns:
        Dict containing:
            - state: "SUCCESS" or "FAILURE"
            - message: Status/error message
            - data_name: Input table name
            - topic_word_scores: Dictionary mapping topics to word scores 
            - topic_doc_scores: Document-topic distribution scores
            - coherence_scores: Topic coherence metrics
            - topic_dist_img: Topic distribution visualization (if enabled)
            - topic_document_counts: Topic size distribution
            - topic_relationships: Inter-topic relationship scores
            
    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If required options are missing or invalid
        KeyError: If desired column is not found in input data
    """
    # Create console manager if not provided
    if console is None:
        console = ConsoleManager()
        
    try:
        _validate_inputs(filepath, desired_columns, options)
        
        # Setup stage
        setup_start = time.time()
        console.print_status(f"Setting up analysis for {table_name}", "processing")
        db_config = DatabaseManager.initialize_database_config(output_base_dir)
        output_dir = db_config.output_dir
        desired_columns = desired_columns.strip() if desired_columns else None
        console.record_stage_time("Setup", setup_start)

        # Use progress context for the main analysis pipeline
        with console.progress_context("Topic Analysis Pipeline") as progress_console:
            # Data loading stage
            data_start = time.time()
            data_task = progress_console.add_task("Loading and preprocessing data", total=3)
            
            df = _load_data_file(filepath, options, progress_console)
            progress_console.update_task(data_task)
            
            df = _preprocess_dataframe(df, desired_columns, options, db_config.main_db_engine, table_name, progress_console)
            progress_console.update_task(data_task)
            
            tdm, vocab, counterized_data, text_array, options = _perform_text_processing(df, desired_columns, options, progress_console)
            progress_console.complete_task(data_task, "Data loading and preprocessing completed")
            console.record_stage_time("Data Loading & Preprocessing", data_start)

            # Topic modeling stage
            modeling_start = time.time()
            modeling_task = progress_console.add_task("Performing topic modeling", total=2)
            
            table_output_dir = output_dir / table_name
            table_output_dir.mkdir(parents=True, exist_ok=True)

            topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result = _perform_topic_modeling(
                tdm, options, vocab, text_array, df, desired_columns, db_config, table_name, table_output_dir, progress_console
            )
            progress_console.update_task(modeling_task)
            console.record_stage_time("NMF Topic Modeling", modeling_start)

            # Output generation stage
            output_start = time.time()
            visual_returns = _generate_outputs(
                nmf_output, vocab, table_output_dir, table_name, options, 
                word_result, topic_word_scores, text_array, db_config.topics_db_engine, 
                db_config.program_output_dir, output_dir, topic_doc_scores, progress_console
            )
            progress_console.complete_task(modeling_task, "Topic modeling and output generation completed")
            console.record_stage_time("Output Generation", output_start)

        console.print_status("Analysis completed successfully!", "success")

        return {
            "state": "SUCCESS",
            "message": "Topic modeling completed successfully",
            "data_name": table_name,
            "topic_word_scores": topic_word_scores,
            "topic_doc_scores": topic_doc_scores,
            "coherence_scores": coherence_scores,
            "topic_dist_img": visual_returns[0] if options["gen_topic_distribution"] else None,
            "topic_document_counts": visual_returns[1] if options["gen_topic_distribution"] else None,
            "topic_relationships": nmf_output.get("S", None),
        }

    except Exception as e:
        console.print_status(f"Analysis failed: {str(e)}", "error")
        return {"state": "FAILURE", "message": str(e), "data_name": table_name}


def run_manta_process(
    filepath,
    table_name: str, 
    desired_columns: str, 
    options: Dict[str, Any], 
    output_base_dir: Optional[str] = None
) -> Dict[str, Any]:

    """
    Main entry point for standalone NMF topic modeling.
    
    Initializes tokenizer and emoji processing, then calls process_file().
    """
    # Initialize console manager and timing
    console = ConsoleManager()
    console.start_timing()
    
    # Display header and configuration
    console.print_header(
        "MANTA Topic Analysis",
        "Multi-lingual Advanced NMF-based Topic Analysis"
    )
    console.display_config(options, filepath, desired_columns, table_name)
    
    console.print_status("Initializing analysis components...", "processing")
    
    init_start = time.time()
    if not options.get("tokenizer"):
        options["tokenizer"] = init_tokenizer(tokenizer_type=options["tokenizer_type"])
    
    options["emoji_map"] = EmojiMap() if options.get("emoji_map") else None
    console.record_stage_time("Initialization", init_start)

    result = process_file(filepath, table_name, desired_columns, options, output_base_dir, console)

    total_time = console.get_total_time()
    console.print_analysis_summary(result, console.stage_times, total_time)
    
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
        },
        "save_to_db": False
    }
    run_manta_process(filepath, table_name, desired_columns, options)
