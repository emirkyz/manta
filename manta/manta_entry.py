"""Main entry point for MANTA topic analysis."""

import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sparse

from ._functions.common_language.emoji_processor import EmojiMap
from .utils.database.database_manager import DatabaseManager
from .utils.console.console_manager import ConsoleManager
from .utils.processing_utils import ProcessingPaths, CachedData, ModelComponents
from .utils.cache_manager import CacheManager
from .pipeline import DataPipeline, TextPipeline, ModelingPipeline, OutputPipeline


def setup_processing(
        table_name: str,
        preprocessing_name: str,
        output_base_dir: Optional[str],
        console: ConsoleManager
) -> Tuple[ProcessingPaths, Any]:
    """Initialize processing paths and database configuration.

    Args:
        table_name: Unique identifier for this analysis run (includes topic count)
        preprocessing_name: Identifier for preprocessing cache (excludes topic count)
        output_base_dir: Base directory for outputs (optional)
        console: Console manager for status messages

    Returns:
        Tuple of (ProcessingPaths, DatabaseConfig)
    """
    console.print_status(f"Setting up analysis for {table_name}", "processing")
    setup_start = time.time()

    # Initialize database configuration
    db_config = DatabaseManager.initialize_database_config(output_base_dir)

    # Create processing paths manager
    paths = ProcessingPaths(
        output_dir=db_config.output_dir,
        table_name=table_name,
        preprocessing_name=preprocessing_name
    )

    console.record_stage_time("Setup", setup_start)
    return paths, db_config


def load_or_process_data(
        filepath: Optional[str],
        dataframe: Optional[pd.DataFrame],
        desired_columns: str,
        options: Dict[str, Any],
        paths: ProcessingPaths,
        db_config: Any,
        console: ConsoleManager
) -> CachedData:
    """Load data from cache or process from file/DataFrame.

    This function handles the data loading and preprocessing pipeline,
    with intelligent caching to skip expensive operations when possible.

    If cached data exists and use_cache=True, the user will be prompted
    interactively to choose whether to use the cache or reprocess.
    Set force_reprocess=True to skip the prompt and always reprocess.

    Args:
        filepath: Path to input file (optional if dataframe provided)
        dataframe: Pre-loaded DataFrame (optional if filepath provided)
        desired_columns: Column name containing text to analyze
        options: Processing configuration options (use_cache, force_reprocess)
        paths: Processing paths manager
        db_config: Database configuration
        console: Console manager

    Returns:
        CachedData object with TF-IDF matrix and metadata
    """
    data_start = time.time()

    # Initialize emoji_map if needed (before cache check)
    # This ensures it's properly initialized whether we use cache or not
    if options.get("emoji_map") is True:
        options["emoji_map"] = EmojiMap()
    elif not options.get("emoji_map"):
        options["emoji_map"] = None

    # Check if cache exists
    use_cache = options.get("use_cache", True)
    force_reprocess = options.get("force_reprocess", False)
    skip_data_loading = "n"

    if paths.cache_exists() and use_cache and not force_reprocess:
        console.print_status(
            f"TF-IDF matrix and metadata already exist ({paths.tfidf_matrix_file.name}, {paths.metadata_file.name})",
            "info"
        )

        # Ask user interactively if they want to skip data loading
        skip_data_loading = input(
            "Do you want to skip data loading and preprocessing? (y/n): "
        ).strip().lower()

        if skip_data_loading == 'y':
            console.print_status("Skipping data loading and preprocessing. Loading files...", "info")

            try:
                # Load from cache
                cached_data = CacheManager.load_cached_data(paths, console)
                console.record_stage_time("Data Loading (from cache)", data_start)
                return cached_data

            except Exception as e:
                console.print_status(
                    f"Failed to load cache: {e}. Re-processing data...",
                    "warning"
                )
                skip_data_loading = "n"
        else:
            console.print_status("Pre-processed data will not be loaded", "info")

    # Process data from scratch
    console.print_status("Processing data from source", "processing")

    # Load dataframe
    df = DataPipeline.load_data(filepath, dataframe, options, console)

    # Preprocess dataframe
    df = DataPipeline.preprocess_dataframe(
        df, desired_columns, options,
        db_config.main_db_engine, paths.table_name, console
    )

    # Extract datetime series if available for temporal analysis
    datetime_series = None
    if options.get('datetime_column') and options['datetime_column'] in df.columns:
        datetime_col = df[options['datetime_column']]
        # Keep as datetime type for proper handling in visualization
        # (avoid conversion to POSIX timestamp which causes incorrect date interpretation)
        datetime_series = datetime_col.copy()

        console.print_status(
            f"Extracted {len(datetime_series)} datetime values for temporal analysis",
            "info"
        )

    # Perform text processing to create TF-IDF matrix
    tdm, vocab, counterized_data, text_array, options = TextPipeline.perform_text_processing(
        df, desired_columns, options, console
    )

    # Print final data statistics summary
    print(f"\n{'='*60}")
    print(f"FINAL DATA STATISTICS")
    print(f"{'='*60}")
    print(f"  DataFrame rows: {len(df)}")
    print(f"  Text array length: {len(text_array)}")
    print(f"  Non-empty texts: {sum(1 for t in text_array if t and t.strip())}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  TF-IDF matrix shape: {tdm.shape}")
    print(f"    Documents (rows): {tdm.shape[0]}")
    print(f"    Features (columns): {tdm.shape[1]}")
    if datetime_series is not None:
        print(f"  Datetime values: {len(datetime_series)}")
    print(f"{'='*60}\n")

    # Create cached data object
    cached_data = CachedData(
        tdm=tdm,
        vocab=vocab,
        text_array=text_array,
        datetime_series=datetime_series
    )

    # Save to cache for future use
    CacheManager.save_cached_data(paths, cached_data, console)

    console.record_stage_time("Data Loading & Preprocessing", data_start)
    return cached_data


def process_file(
        filepath: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        table_name: str = None,
        desired_columns: str = None,
        options: Dict[str, Any] = None,
        output_base_dir: Optional[str] = None,
        console: Optional[ConsoleManager] = None,
) -> Dict[str, Any]:
    """Process a file or DataFrame and perform NMF topic modeling analysis.

    This is the main entry point for topic modeling. It handles the complete
    pipeline including data loading, preprocessing, NMF analysis, and output
    generation with support for both file and DataFrame inputs.

    Args:
        filepath: Path to input CSV or Excel file (optional if dataframe provided)
        dataframe: Pre-loaded pandas DataFrame (optional if filepath provided)
        table_name: Unique identifier used for naming output files
        desired_columns: Column name containing the text to analyze
        options: Dictionary containing processing configuration parameters
        output_base_dir: Base directory for outputs (optional, defaults to current dir)
        console: Console manager for status messages (optional)

    Returns:
        Dict containing:
            - state: "SUCCESS" or "FAILURE"
            - message: Status/error message
            - data_name: Input table name

    Raises:
        ValueError: If both or neither filepath and dataframe are provided
        FileNotFoundError: If input file does not exist
        KeyError: If desired column is not found in input data
    """
    # Create console manager if not provided
    if console is None:
        console = ConsoleManager()

    try:
        # Validate inputs
        DataPipeline.validate_inputs(filepath, desired_columns, options, dataframe)

        # Setup processing paths and database config
        desired_columns = desired_columns.strip() if desired_columns else None

        # Create preprocessing name (without NMF type and topic count)
        # Preprocessing is independent of modeling parameters
        # table_name format: {data_name}_{nmf_type}_{tokenizer_type}_{topic_count}
        # preprocessing_name should be: {data_name}_{tokenizer_type}
        parts = table_name.rsplit('_', 3)  # Split from right, max 3 splits
        if len(parts) == 4:
            # Successfully split into [data_name, nmf_type, tokenizer_type, topic_count]
            data_name = parts[0]
            tokenizer_type = parts[2]
            preprocessing_name = f"{data_name}_{tokenizer_type}"
        else:
            # Fallback: use table_name as-is if format doesn't match expected pattern
            preprocessing_name = table_name

        paths, db_config = setup_processing(table_name, preprocessing_name, output_base_dir, console)

        # Load or process data (with caching support)
        with console.progress_context("Topic Analysis Pipeline"):
            cached_data = load_or_process_data(
                filepath, dataframe, desired_columns, options,
                paths, db_config, console
            )

        # Create output directory
        table_output_dir = paths.table_output_dir(table_name)
        table_output_dir.mkdir(parents=True, exist_ok=True)

        # Topic modeling stage
        modeling_start = time.time()
        console.print_status("Performing topic modeling", "processing")

        topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result = (
            ModelingPipeline.perform_topic_modeling(
                cached_data.tdm, options, cached_data.vocab, cached_data.text_array,
                db_config, table_name, table_output_dir, console, desired_columns
            )
        )
        console.record_stage_time("NMF Topic Modeling", modeling_start)

        # Output generation stage
        output_start = time.time()
        console.print_status("Generating outputs", "processing")

        visual_returns = OutputPipeline.generate_outputs(
            nmf_output, cached_data.vocab, table_output_dir, table_name, options,
            word_result, topic_word_scores, cached_data.text_array, db_config.topics_db_engine,
            db_config.program_output_dir, db_config.output_dir, topic_doc_scores, console,
            datetime_series=cached_data.datetime_series
        )
        console.record_stage_time("Output Generation", output_start)

        # Save model components
        model_components = ModelComponents.from_nmf_output(
            nmf_output, cached_data.vocab, cached_data.text_array
        )
        CacheManager.save_model_components(paths, model_components, table_name, console)

        return {
            "state": "SUCCESS",
            "message": "Topic modeling completed successfully",
            "data_name": table_name,
            "topic_word_scores": topic_word_scores,
            "topic_doc_scores": topic_doc_scores,
            "coherence_scores": coherence_scores,
            "visual_returns": visual_returns
        }

    except Exception as e:
        console.print_status(f"Analysis failed: {str(e)}", "error")
        return {
            "state": "FAILURE",
            "message": str(e),
            "data_name": table_name
        }


def run_manta_process(
        filepath: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        table_name: str = None,
        desired_columns: str = None,
        options: Dict[str, Any] = None,
        output_base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Main entry point for MANTA topic modeling.

    Initializes emoji processing and console, then calls process_file().
    Supports both file and DataFrame inputs.

    Args:
        filepath: Path to input file (optional if dataframe provided)
        dataframe: Pre-loaded DataFrame (optional if filepath provided)
        table_name: Unique identifier for this analysis
        desired_columns: Column name containing text to analyze
        options: Processing configuration
        output_base_dir: Base directory for outputs

    Returns:
        Dictionary with analysis results

    Raises:
        ValueError: If both or neither filepath and dataframe are provided
    """
    # Validate that exactly one input is provided
    if filepath is None and dataframe is None:
        raise ValueError("Either filepath or dataframe must be provided")
    if filepath is not None and dataframe is not None:
        raise ValueError("Cannot provide both filepath and dataframe - choose one")

    # Initialize console manager and timing
    console = ConsoleManager()
    console.start_timing()

    # Display header and configuration
    console.print_header(
        "MANTA Topic Analysis",
        "Multi-lingual Advanced NMF-based Topic Analysis"
    )

    # Show input source in config
    if filepath:
        console.display_config(options, filepath, desired_columns, table_name)
    else:
        console.print_status(f"Input: DataFrame with {len(dataframe)} rows", "info")
        console.display_config(options, None, desired_columns, table_name)

    console.print_status("Initializing analysis components...", "processing")

    init_start = time.time()
    console.record_stage_time("Initialization", init_start)

    # Run the main processing pipeline
    result = process_file(
        filepath=filepath,
        dataframe=dataframe,
        table_name=table_name,
        desired_columns=desired_columns,
        options=options,
        output_base_dir=output_base_dir,
        console=console
    )

    # Display summary
    total_time = console.get_total_time()
    console.print_analysis_summary(result, console.stage_times, total_time)

    return result


if __name__ == "__main__":
    # Example configuration
    LEMMATIZE = True
    N_WORDS = 15
    DESIRED_TOPIC_COUNT = 5
    tokenizer_type = "bpe"  # "wordpiece" or "bpe"
    nmf_type = "nmf"
    filepath = "veri_setleri/APPSTORE_APP_REVIEWSyeni_yeni.csv"
    data_name = filepath.split("/")[-1].split(".")[0].split("_")[0]
    LANGUAGE = "TR"
    separator = "|"
    table_name = f"{data_name}_{nmf_type}_{tokenizer_type}_{DESIRED_TOPIC_COUNT}"
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
        "filter_app": False,
        "data_filter_options": {
            "filter_app_name": "",
            "filter_app_column": "PACKAGE_NAME",
            "filter_app_country": "TR",
            "filter_app_country_column": "COUNTRY",
        },
        "save_to_db": False,
        "use_cache": True,  # Use cached data if available
        "force_reprocess": False,  # Set to True to force reprocessing
        "nmf_variants": ["nmf"],  # Can run multiple variants: ["nmf", "nmtf"]
    }

    run_manta_process(filepath=filepath, table_name=table_name,
                      desired_columns=desired_columns, options=options)
