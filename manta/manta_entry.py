"""Main entry point for MANTA topic analysis."""

import time
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd

from ._functions.common_language.emoji_processor import EmojiMap
from .config import TopicAnalysisConfig
from .utils.database.database_manager import DatabaseManager
from .utils.console.console_manager import ConsoleManager
from .utils.processing_utils import ProcessingPaths, CachedData, ModelComponents, PipelineContext
from .utils.cache_manager import CacheManager
from .pipeline import DataPipeline, TextPipeline, ModelingPipeline, OutputPipeline


def _setup_stage(
    config: TopicAnalysisConfig,
    column: str,
    output_base_dir: Optional[str],
    console: ConsoleManager,
) -> PipelineContext:
    """Create paths and db config, return bundled PipelineContext."""
    table_name = config.output_name
    parts = table_name.rsplit("_", 3)
    preprocessing_name = f"{parts[0]}_{parts[2]}" if len(parts) == 4 else table_name

    console.print_status(f"Setting up analysis for {table_name}", "processing")
    setup_start = time.time()

    db_config = DatabaseManager.initialize_database_config(output_base_dir)
    paths = ProcessingPaths(
        output_dir=db_config.output_dir,
        table_name=table_name,
        preprocessing_name=preprocessing_name,
    )

    console.record_stage_time("Setup", setup_start)
    return PipelineContext(paths=paths, db_config=db_config, console=console, column=column)


def _data_stage(
    filepath: Optional[str],
    dataframe: Optional[pd.DataFrame],
    compat_options: Dict[str, Any],
    ctx: PipelineContext,
) -> CachedData:
    """Load or process data. Populates ctx.emoji_map and ctx.tokenizer as a side effect."""
    data_start = time.time()

    # Initialize emoji_map before cache check so it's ready for both paths
    if compat_options.get("emoji_map") is True:
        compat_options["emoji_map"] = EmojiMap()
    elif not compat_options.get("emoji_map"):
        compat_options["emoji_map"] = None
    ctx.emoji_map = compat_options["emoji_map"]

    use_cache = compat_options.get("use_cache", True)
    force_reprocess = compat_options.get("force_reprocess", False)

    if ctx.paths.cache_exists() and use_cache and not force_reprocess and not compat_options.get("barebones"):
        ctx.console.print_status(
            f"TF-IDF matrix and metadata already exist "
            f"({ctx.paths.tfidf_matrix_file.name}, {ctx.paths.metadata_file.name})",
            "info",
        )
        skip = input("Do you want to skip data loading and preprocessing? (y/n): ").strip().lower()
        if skip == "y":
            ctx.console.print_status("Skipping data loading and preprocessing. Loading files...", "info")
            try:
                cached_data = CacheManager.load_cached_data(ctx.paths, ctx.console)
                if cached_data.datetime_is_combined:
                    compat_options["datetime_is_combined_year_month"] = True
                ctx.console.record_stage_time("Data Loading (from cache)", data_start)
                return cached_data
            except Exception as e:
                ctx.console.print_status(f"Failed to load cache: {e}. Re-processing data...", "warning")
        else:
            ctx.console.print_status("Pre-processed data will not be loaded", "info")

    ctx.console.print_status("Processing data from source", "processing")

    df = DataPipeline.load_data(filepath, dataframe, compat_options, ctx.console)
    df = DataPipeline.preprocess_dataframe(
        df, ctx.column, compat_options,
        ctx.db_config.main_db_engine, ctx.paths.table_name, ctx.console,
    )

    datetime_series = None
    datetime_info = compat_options.get("_datetime_info")
    if datetime_info and compat_options.get("datetime_column") and compat_options["datetime_column"] in df.columns:
        from manta.utils.datetime_handler import DatetimeDetector
        raw_col = df[compat_options["datetime_column"]].copy()
        datetime_series = DatetimeDetector.convert_to_datetime(raw_col, datetime_info.format)
        ctx.console.print_status(
            f"Extracted {len(datetime_series)} datetime values for temporal analysis", "info"
        )

    tdm, vocab, _, text_array, original_text_array, datetime_series, compat_options = TextPipeline.perform_text_processing(
        df, ctx.column, compat_options, ctx.console, datetime_series=datetime_series
    )

    if not compat_options.get("use_original_data", True):
        original_text_array = text_array.copy()

    # Capture runtime state produced by text processing into the context
    ctx.emoji_map = compat_options.get("emoji_map")
    ctx.tokenizer = compat_options.get("tokenizer")

    ctx.console.print_debug("=" * 60, tag="DATA STATISTICS")
    ctx.console.print_debug("FINAL DATA STATISTICS", tag="DATA STATISTICS")
    ctx.console.print_debug("=" * 60, tag="DATA STATISTICS")
    ctx.console.print_debug(f"  DataFrame rows: {len(df)}", tag="DATA STATISTICS")
    ctx.console.print_debug(f"  Text array length: {len(text_array)}", tag="DATA STATISTICS")
    ctx.console.print_debug(f"  Non-empty texts: {sum(1 for t in text_array if t and t.strip())}", tag="DATA STATISTICS")
    ctx.console.print_debug(f"  Vocabulary size: {len(vocab)}", tag="DATA STATISTICS")
    ctx.console.print_debug(f"  TF-IDF matrix shape: {tdm.shape}", tag="DATA STATISTICS")
    if datetime_series is not None:
        ctx.console.print_debug(f"  Datetime values: {len(datetime_series)}", tag="DATA STATISTICS")
    ctx.console.print_debug("=" * 60, tag="DATA STATISTICS")

    cached_data = CachedData(
        tdm=tdm,
        vocab=vocab,
        text_array=text_array,
        original_text_array=original_text_array,
        datetime_series=datetime_series,
        datetime_is_combined=compat_options.get("datetime_is_combined_year_month", False),
        pagerank_weights=compat_options.get("pagerank_weights"),
    )

    CacheManager.save_cached_data(ctx.paths, cached_data, ctx.console)
    ctx.console.record_stage_time("Data Loading & Preprocessing", data_start)
    return cached_data


def _model_stage(
    cached_data: CachedData,
    config: TopicAnalysisConfig,
    ctx: PipelineContext,
) -> Tuple[Dict, Dict, Dict, Dict, Any]:
    """Run NMF topic modeling. Returns (topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result)."""
    modeling_start = time.time()
    ctx.console.print_status("Performing topic modeling", "processing")

    table_output_dir = ctx.paths.table_output_dir(ctx.paths.table_name)
    table_output_dir.mkdir(parents=True, exist_ok=True)

    result = ModelingPipeline.perform_topic_modeling(
        cached_data=cached_data,
        config=config,
        ctx=ctx,
        table_output_dir=table_output_dir,
    )

    ctx.console.record_stage_time("NMF Topic Modeling", modeling_start)
    return result


def _output_stage(
    cached_data: CachedData,
    nmf_output: Dict,
    word_result: Any,
    topic_word_scores: Dict,
    topic_doc_scores: Dict,
    config: TopicAnalysisConfig,
    ctx: PipelineContext,
) -> Any:
    """Generate outputs and save model components. Returns visual_returns."""
    output_start = time.time()
    ctx.console.print_status("Generating outputs", "processing")

    table_output_dir = ctx.paths.table_output_dir(ctx.paths.table_name)

    visual_returns = OutputPipeline.generate_outputs(
        cached_data=cached_data,
        nmf_output=nmf_output,
        word_result=word_result,
        topic_word_scores=topic_word_scores,
        topic_doc_scores=topic_doc_scores,
        config=config,
        ctx=ctx,
        table_output_dir=table_output_dir,
    )

    model_components = ModelComponents.from_nmf_output(nmf_output, cached_data.vocab, cached_data.text_array)
    CacheManager.save_model_components(ctx.paths, model_components, ctx.paths.table_name, ctx.console)

    ctx.console.record_stage_time("Output Generation", output_start)
    return visual_returns


def process_file(
    config: TopicAnalysisConfig,
    filepath: Optional[str] = None,
    dataframe: Optional[pd.DataFrame] = None,
    column: str = None,
    output_base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete topic analysis pipeline.

    Args:
        config: Typed analysis configuration (single source of truth for all settings)
        filepath: Path to input file (optional if dataframe provided)
        dataframe: Pre-loaded DataFrame (optional if filepath provided)
        column: Column name containing text to analyze
        output_base_dir: Base directory for outputs (defaults to current directory)

    Returns:
        Dict with state, message, and analysis results
    """
    console = ConsoleManager()
    console.start_timing()
    console.print_header("MANTA Topic Analysis", "Multi-lingual Advanced NMF-based Topic Analysis")

    # Build compat options dict once — used only for display and DataPipeline/TextPipeline
    compat_options = config.to_run_options()

    if filepath:
        console.display_config(compat_options, filepath, column, config.output_name)
    else:
        console.print_status(f"Input: DataFrame with {len(dataframe)} rows", "info")
        console.display_config(compat_options, None, column, config.output_name)

    console.print_status("Initializing analysis components...", "processing")
    init_start = time.time()
    console.record_stage_time("Initialization", init_start)

    try:
        # Validate runtime inputs (config fields already validated in TopicAnalysisConfig.__post_init__)
        if filepath is None and dataframe is None:
            raise ValueError("Either filepath or dataframe must be provided")
        if filepath is not None and dataframe is not None:
            raise ValueError("Cannot provide both filepath and dataframe")
        if filepath is not None and not Path(filepath).exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")
        if dataframe is not None and (not isinstance(dataframe, pd.DataFrame) or dataframe.empty):
            raise ValueError("dataframe must be a non-empty DataFrame")

        column = column.strip() if column else None
        if not column:
            raise ValueError("column cannot be empty")

        ctx = _setup_stage(config, column, output_base_dir, console)

        with console.progress_context("Topic Analysis Pipeline"):
            cached_data = _data_stage(filepath, dataframe, compat_options, ctx)

        topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result = (
            _model_stage(cached_data, config, ctx)
        )

        if not config.barebones:
            visual_returns = _output_stage(
                cached_data, nmf_output, word_result,
                topic_word_scores, topic_doc_scores, config, ctx,
            )
        else:
            visual_returns = None

        from .utils.analysis.dominant_topic import get_dominant_topics
        dominant_topics = get_dominant_topics(nmf_output["W"])
        document_dominant_topics = {int(i): int(t) for i, t in enumerate(dominant_topics)}

        result = {
            "state": "SUCCESS",
            "message": "Topic modeling completed successfully",
            "data_name": config.output_name,
            "topic_word_scores": topic_word_scores,
            "topic_doc_scores": topic_doc_scores,
            "coherence_scores": coherence_scores,
            "visual_returns": visual_returns,
            "document_dominant_topics": document_dominant_topics,
        }

        if config.barebones:
            result["W"] = nmf_output["W"]
            result["H"] = nmf_output["H"]
            result["vocab"] = cached_data.vocab

    except Exception as e:
        console.print_status(f"Analysis failed: {str(e)}", "error")
        result = {
            "state": "FAILURE",
            "message": str(e),
            "data_name": config.output_name if config else None,
        }

    total_time = console.get_total_time()
    console.print_analysis_summary(result, console.stage_times, total_time)
    return result


def run_manta_process(
    filepath: Optional[str] = None,
    dataframe: Optional[pd.DataFrame] = None,
    table_name: str = None,
    desired_columns: str = None,
    options: Dict[str, Any] = None,
    output_base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Backward-compatible entry point. Prefer calling process_file with TopicAnalysisConfig directly."""
    from .config import create_config_from_params, DataFilterOptions

    if options is None:
        options = {}

    data_filter_opts = options.get("data_filter_options", {})
    if not isinstance(data_filter_opts, dict):
        data_filter_opts = data_filter_opts.__dict__ if hasattr(data_filter_opts, "__dict__") else {}

    emoji_raw = options.get("emoji_map", False)
    emoji_bool = bool(emoji_raw) if not hasattr(emoji_raw, "process") else True

    config = create_config_from_params(
        language=options.get("LANGUAGE", "EN"),
        topic_count=options.get("DESIRED_TOPIC_COUNT", 5),
        words_per_topic=options.get("N_TOPICS", 15),
        nmf_method=options.get("nmf_type", "nmf"),
        tokenizer_type=options.get("tokenizer_type", "bpe"),
        lemmatize=options.get("LEMMATIZE", False),
        generate_wordclouds=options.get("gen_cloud", True),
        export_excel=options.get("save_excel", True),
        topic_distribution=options.get("gen_topic_distribution", True),
        separator=options.get("separator", ","),
        filter_app=options.get("filter_app", False),
        data_filter_options=data_filter_opts or None,
        emoji_map=emoji_bool,
        word_pairs_out=options.get("word_pairs_out", True),
        save_to_db=options.get("save_to_db", False),
        n_grams_to_discover=options.get("n_grams_to_discover"),
        ngram_auto_k=options.get("ngram_auto_k", 0.5),
        keep_numbers=options.get("keep_numbers", False),
        filter_standalone_numbers=options.get("filter_standalone_numbers", True),
        use_pmi=options.get("use_pmi", True),
        use_cache=options.get("use_cache", True),
        force_reprocess=options.get("force_reprocess", False),
        output_name=table_name,
        pagerank_column=options.get("pagerank_column"),
        datetime_column=options.get("datetime_column"),
        time_grouping=options.get("time_grouping"),
    )

    return process_file(
        config=config,
        filepath=filepath,
        dataframe=dataframe,
        column=desired_columns,
        output_base_dir=output_base_dir,
    )


if __name__ == "__main__":
    from .config import create_config_from_params

    config = create_config_from_params(
        language="TR",
        topic_count=5,
        nmf_method="nmf",
        tokenizer_type="bpe",
        lemmatize=True,
        words_per_topic=15,
        separator="|",
        generate_wordclouds=True,
        export_excel=True,
        word_pairs_out=True,
        topic_distribution=True,
        emoji_map=True,
        output_name="APPSTORE_nmf_bpe_5",
    )

    process_file(
        config=config,
        filepath="veri_setleri/APPSTORE_APP_REVIEWSyeni_yeni.csv",
        column="REVIEW",
    )
