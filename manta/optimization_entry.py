"""Main entry point for MANTA topic count optimization."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ._functions.common_language.emoji_processor import EmojiMap
from .config import OptimizationConfig
from .pipeline import DataPipeline, TextPipeline
from .pipeline.optimization_pipeline import OptimizationPipeline, OptimizationResult
from .utils.cache_manager import CacheManager
from .utils.console.console_manager import ConsoleManager
from .utils.database.database_manager import DatabaseManager
from .utils.export.optimization_results import (
    generate_optimization_summary,
    save_optimization_results,
)
from .utils.processing_utils import CachedData, ProcessingPaths
from .utils.visualization.coherence_plot import plot_coherence_results


def run_optimization_process(
    filepath: Optional[str] = None,
    dataframe: Optional[pd.DataFrame] = None,
    column: str = None,
    config: Optional[OptimizationConfig] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point for topic count optimization.

    Finds the optimal number of topics by:
    1. Preprocessing data once (TF-IDF matrix creation)
    2. Running NMF with different topic counts
    3. Calculating coherence scores for each
    4. Applying elbow detection for alternative recommendation
    5. Generating plots and saving results

    Args:
        filepath: Path to input file (optional if dataframe provided).
        dataframe: Pre-loaded DataFrame (optional if filepath provided).
        column: Column name containing text to analyze.
        config: OptimizationConfig with all settings.
        output_dir: Base directory for outputs.

    Returns:
        Dict with optimization results including:
            - state: "SUCCESS" or "FAILURE"
            - optimal_topic_count: Recommended number of topics
            - optimal_coherence: Coherence score at optimal
            - elbow_topic_count: Alternative recommendation from elbow detection
            - results: Full results dictionary
            - output_dir: Path to output files

    Raises:
        ValueError: If both or neither filepath and dataframe are provided.
    """
    # Validate inputs
    if filepath is None and dataframe is None:
        raise ValueError("Either filepath or dataframe must be provided")
    if filepath is not None and dataframe is not None:
        raise ValueError("Cannot provide both filepath and dataframe - choose one")

    # Use default config if not provided
    if config is None:
        config = OptimizationConfig()

    # Initialize console manager and timing
    console = ConsoleManager()
    console.start_timing()

    # Display header
    console.print_header(
        "MANTA Topic Count Optimization",
        f"Finding optimal topic count using {config.nmf_method.upper()} method",
    )

    # Display configuration
    _display_optimization_config(config, filepath, column, console)

    try:
        # Setup paths
        preprocessing_name = _generate_preprocessing_name(filepath, dataframe, config)
        db_config = DatabaseManager.initialize_database_config(output_dir)

        paths = ProcessingPaths(
            output_dir=db_config.output_dir,
            table_name=f"{preprocessing_name}_optimization",
            preprocessing_name=preprocessing_name,
        )

        # Step 1: Preprocess data
        preprocessing_start = time.time()
        console.print_header("Data Loading & Preprocessing")

        options = config.to_text_processing_options()

        # Initialize emoji_map if needed
        if options.get("emoji_map") is True:
            options["emoji_map"] = EmojiMap()
        elif not options.get("emoji_map"):
            options["emoji_map"] = None

        # Check for cached data
        cached_data = None
        if paths.cache_exists() and config.use_cache and not config.force_reprocess:
            console.print_status("Found cached TF-IDF matrix", "info")
            try:
                cached_data = CacheManager.load_cached_data(paths, console)
                console.print_status("Loaded cached preprocessing data", "success")
            except Exception as e:
                console.print_status(
                    f"Cache load failed: {e}. Reprocessing...", "warning"
                )
                cached_data = None

        if cached_data is None:
            # Load and preprocess data
            console.print_status("Processing data from source", "processing")

            df = DataPipeline.load_data(filepath, dataframe, options, console)
            df = DataPipeline.preprocess_dataframe(
                df,
                column,
                options,
                db_config.main_db_engine,
                preprocessing_name,
                console,
            )

            # Text processing to create TF-IDF
            tdm, vocab, _, text_array, original_test_temp, options = TextPipeline.perform_text_processing(
                df, column, options, console
            )

            cached_data = CachedData(
                tdm=tdm,
                vocab=vocab,
                text_array=text_array,
            )

            # Save cache for future use
            CacheManager.save_cached_data(paths, cached_data, console)

        preprocessing_time = time.time() - preprocessing_start
        console.print_status(
            f"Preprocessing completed in {preprocessing_time:.2f}s", "success"
        )
        console.print_status(
            f"TF-IDF matrix: {cached_data.tdm.shape[0]} docs x {cached_data.tdm.shape[1]} features",
            "info",
        )

        # Step 2: Run optimization
        topic_counts = config.get_topic_counts()
        console.print_status(
            f"Evaluating {len(topic_counts)} topic counts: {topic_counts[0]} to {topic_counts[-1]}",
            "info",
        )

        result = OptimizationPipeline.run_optimization(
            tfidf_matrix=cached_data.tdm,
            vocab=cached_data.vocab,
            text_array=cached_data.text_array,
            topic_counts=topic_counts,
            nmf_method=config.nmf_method,
            words_per_topic=config.words_per_topic,
            lambda_val=config.lambda_val,
            console=console,
        )
        result.preprocessing_time = preprocessing_time

        # Step 3: Generate outputs
        output_path = paths.table_output_dir()
        output_path.mkdir(parents=True, exist_ok=True)

        # Find indices for plotting
        optimal_idx = result.topic_counts.index(result.optimal_topic_count)
        elbow_idx = None
        if result.elbow_topic_count is not None:
            elbow_idx = result.topic_counts.index(result.elbow_topic_count)

        if config.save_plot or config.show_plot:
            plot_path = plot_coherence_results(
                topic_counts=result.topic_counts,
                coherence_scores=result.coherence_scores,
                output_dir=str(output_path),
                nmf_method=config.nmf_method,
                optimal_idx=optimal_idx,
                elbow_idx=elbow_idx,
                save_plot=config.save_plot,
                show_plot=config.show_plot,
            )
            if plot_path:
                console.print_status(f"Plot saved to: {plot_path}", "success")

        if config.save_csv or config.save_json:
            saved_files = save_optimization_results(
                result=result,
                output_dir=str(output_path),
                save_csv=config.save_csv,
                save_json=config.save_json,
            )
            for file_type, file_path in saved_files.items():
                if file_path:
                    console.print_status(
                        f"{file_type.upper()} saved to: {file_path}", "success"
                    )

        # Print summary
        total_time = console.get_total_time()
        _print_optimization_summary(result, total_time, output_path, console)

        # Print text summary
        summary_text = generate_optimization_summary(result)
        print(summary_text)

        return {
            "state": "SUCCESS",
            "optimal_topic_count": result.optimal_topic_count,
            "optimal_coherence": result.optimal_coherence,
            "elbow_topic_count": result.elbow_topic_count,
            "results": result.to_dict(),
            "output_dir": str(output_path),
        }

    except Exception as e:
        console.print_status(f"Optimization failed: {str(e)}", "error")
        import traceback

        traceback.print_exc()
        return {"state": "FAILURE", "message": str(e)}


def _display_optimization_config(
    config: OptimizationConfig,
    filepath: Optional[str],
    column: str,
    console: ConsoleManager,
) -> None:
    """Display optimization configuration."""
    console.print_status(f"Input: {filepath or 'DataFrame'}", "info")
    console.print_status(f"Column: {column}", "info")
    console.print_status(
        f"Topic Range: {config.min_topics} to {config.max_topics} (step: {config.step})",
        "info",
    )
    console.print_status(f"NMF Method: {config.nmf_method.upper()}", "info")
    console.print_status(f"Language: {config.language}", "info")
    console.print_status(f"Tokenizer: {config.tokenizer_type}", "info")
    if config.pagerank_column:
        console.print_status(f"PageRank Column: {config.pagerank_column}", "info")


def _generate_preprocessing_name(
    filepath: Optional[str],
    dataframe: Optional[pd.DataFrame],
    config: OptimizationConfig,
) -> str:
    """Generate preprocessing cache name."""
    if filepath:
        base_name = Path(filepath).stem
    else:
        base_name = "dataframe"
    return f"{base_name}_{config.tokenizer_type}"


def _print_optimization_summary(
    result: OptimizationResult,
    total_time: float,
    output_path: Path,
    console: ConsoleManager,
) -> None:
    """Print optimization summary to console."""
    console.print_header("Optimization Results")
    console.print_status(
        f"Optimal Topic Count: {result.optimal_topic_count}", "success"
    )
    console.print_status(
        f"Optimal Coherence Score: {result.optimal_coherence:.4f}", "success"
    )

    if result.elbow_topic_count is not None:
        elbow_idx = result.topic_counts.index(result.elbow_topic_count)
        elbow_coherence = result.coherence_scores[elbow_idx]
        console.print_status(
            f"Elbow Point: {result.elbow_topic_count} topics (C_V: {elbow_coherence:.4f})",
            "info",
        )

    console.print_status(f"Total Time: {total_time:.2f}s", "info")
    console.print_status(f"Preprocessing Time: {result.preprocessing_time:.2f}s", "info")
    console.print_status(f"Optimization Time: {result.total_time:.2f}s", "info")
    console.print_status(f"Output Directory: {output_path}", "info")
