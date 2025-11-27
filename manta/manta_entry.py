import time
from typing import Dict, Any, Optional
import numpy as np
import scipy # Original import
import scipy.sparse as sparse # <-- ADDED for sparse save/load
from ._functions.common_language.emoji_processor import EmojiMap
from ._functions.turkish.turkish_tokenizer_factory import init_tokenizer
import pandas as pd
from .utils.database.database_manager import DatabaseManager
from .utils.console.console_manager import ConsoleManager
from .pipeline import DataPipeline, TextPipeline, ModelingPipeline, OutputPipeline

# All pipeline functions have been moved to separate modules in the pipeline package


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
        DataPipeline.validate_inputs(filepath, desired_columns, options)

        # Setup stage
        setup_start = time.time()
        console.print_status(f"Setting up analysis for {table_name}", "processing")
        db_config = DatabaseManager.initialize_database_config(output_base_dir)
        output_dir = db_config.output_dir
        desired_columns = desired_columns.strip() if desired_columns else None
        console.record_stage_time("Setup", setup_start)

        # Use progress context for the main analysis pipeline
        with console.progress_context("Topic Analysis Pipeline") as progress_console:

            tfidf_matrix_file = output_dir / f"{table_name}_tfidf_matrix.npz"
            metadata_file = output_dir / f"{table_name}_tfidf_metadata.npz"

            # Data loading stage
            data_start = time.time()
            data_task = progress_console.add_task("Loading and preprocessing data", total=3)
            skip_data_loading = "n"

            if tfidf_matrix_file.exists() and metadata_file.exists():
                console.print_status(f"TF-IDF matrix and metadata already exist ({tfidf_matrix_file.name}, {metadata_file.name})", "info")
                skip_data_loading = input(
                    "Do you want to skip data loading and preprocessing? (y/n)")

                if skip_data_loading == 'y':
                    progress_console.complete_task(data_task,
                                                   "Data loading and preprocessing completed. Skipping data loading and preprocessing. Files loaded.")

                    console.print_status(f"Loading sparse matrix from {tfidf_matrix_file.name}...", "processing")
                    tdm = sparse.load_npz(tfidf_matrix_file)

                    console.print_status(f"Loading metadata from {metadata_file.name}...", "processing")
                    with np.load(metadata_file, allow_pickle=True) as file:
                        vocab = list(file["vocab"])
                        text_array = list(file["text_array"])

                        # Handle case where datetime_series might not be in the file
                        datetime_array = file["datetime_series"]

                        # Check if values are years (between 1900-2100) or POSIX timestamps
                        datetime_series = pd.Series(datetime_array)

                        if datetime_series.min() > 1900 and datetime_series.max() < 2100:
                            # These are year values - convert to datetime
                            year_strings = datetime_series.astype(int).astype(str)
                            datetime_series = pd.to_datetime(year_strings, format='%Y')
                        elif datetime_series.min() > 1e9:
                            # These are POSIX timestamps - convert with unit='s'
                            datetime_series = pd.to_datetime(datetime_series, unit='s')

                    console.print_status("Loaded pre-processed data.", "success")

                else:
                    console.print_status("Pre-processed data is not loaded", "info")

            if skip_data_loading=="n" or not tfidf_matrix_file.exists() or not metadata_file.exists():

                df = DataPipeline.load_data_file(filepath, options, progress_console)
                progress_console.update_task(data_task)

                df = DataPipeline.preprocess_dataframe(df, desired_columns, options, db_config.main_db_engine, table_name,
                                                       progress_console)
                progress_console.update_task(data_task)

                # Extract datetime series if available for temporal analysis
                datetime_series = None
                if options.get('datetime_column') and options['datetime_column'] in df.columns:
                    # Convert to integer representation if it's datetime
                    datetime_col = df[options['datetime_column']]
                    if pd.api.types.is_datetime64_any_dtype(datetime_col):
                        # Example: convert to POSIX timestamp (integer)
                        datetime_series = datetime_col.astype('int64') // 10**9
                    else:
                        datetime_series = datetime_col.copy()

                    progress_console.print_status(f"Extracted {len(datetime_series)} datetime values for temporal analysis",
                                                  "info")

                tdm, vocab, counterized_data, text_array, options = TextPipeline.perform_text_processing(df,
                                                                                                         desired_columns,
                                                                                                         options,
                                                                                                         progress_console)
                progress_console.complete_task(data_task, "Data loading and preprocessing completed")
                console.record_stage_time("Data Loading & Preprocessing", data_start)

                console.print_status(f"Saving sparse TF-IDF matrix to {tfidf_matrix_file.name}...", "processing")
                sparse.save_npz(tfidf_matrix_file, tdm)
                console.print_status(f"TF-IDF matrix saved.", "success")

                console.print_status(f"Saving metadata to {metadata_file.name}...", "processing")
                np.savez_compressed(
                    metadata_file,
                    vocab=vocab,
                    text_array=text_array,
                    datetime_series=datetime_series
                )
                console.print_status(f"Metadata saved.", "success")

        nmf_methods_to_corun = ["nmtf"]  # TODO: WILL BE REVERTED. IMPLEMENTED FOR RESEARCH PURPOSES

        for nmf_variant in nmf_methods_to_corun:
            console.print_header(f"NMF variant: {nmf_variant}")

            table_name_variant = table_name.replace(options["nmf_type"], nmf_variant)
            options["nmf_type"] = nmf_variant

            # Create a new progress context for this NMF variant
            with console.progress_context(f"Processing {nmf_variant}") as progress_console:
                modeling_start = time.time()
                modeling_task = progress_console.add_task("Performing topic modeling", total=2)

                table_output_dir = output_dir / table_name_variant
                table_output_dir.mkdir(parents=True, exist_ok=True)

                topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result = ModelingPipeline.perform_topic_modeling(
                    tdm, options, vocab, text_array, db_config, table_name_variant, table_output_dir, progress_console, desired_columns
                )
                progress_console.update_task(modeling_task)
                console.record_stage_time(f"NMF Topic Modeling ({nmf_variant})", modeling_start)


                # Output generation stage
                output_start = time.time()
                visual_returns = OutputPipeline.generate_outputs(
                    nmf_output, vocab, table_output_dir, table_name_variant, options,
                    word_result, topic_word_scores, text_array, db_config.topics_db_engine,
                    db_config.program_output_dir, output_dir, topic_doc_scores, progress_console,
                    datetime_series=datetime_series
                )
                progress_console.complete_task(modeling_task, "Topic modeling and output generation completed")
                console.record_stage_time(f"Output Generation ({nmf_variant})", output_start)

                console.print_status("Saving model components...", "processing")
                model_file = table_output_dir / f"{table_name_variant}_model_components.npz"
                try:
                    # Prepare save data with W, H, vocab, text_array
                    save_data = {
                        'W': nmf_output["W"],
                        'H': nmf_output["H"],
                        'vocab': vocab,
                        'text_array': text_array,
                    }

                    # Add S matrix if present (NMTF only)
                    if "S" in nmf_output:
                        save_data['S'] = nmf_output["S"]

                    np.savez_compressed(model_file, **save_data)
                    console.print_status(f"Model components saved to {model_file.name}", "success")

                except Exception as e:
                    console.print_status(f"Warning: Failed to save model components: {e}", "warning")

        return {
            "state": "SUCCESS",
            "message": f"Topic modeling for all defined NMF variants: {nmf_methods_to_corun} completed successfully",
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
        "filter_app": False,
        "data_filter_options": {
            "filter_app_name": "",
            "filter_app_column": "PACKAGE_NAME",
            "filter_app_country": "TR",
            "filter_app_country_column": "COUNTRY",
        },
        "save_to_db": False
    }
    run_manta_process(filepath, table_name, desired_columns, options)