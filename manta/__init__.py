"""
MANTA (Multi-lingual Advanced NMF-based Topic Analysis) - A comprehensive topic modeling library for Turkish and English texts.

This package provides Non-negative Matrix Factorization (NMF) based topic modeling
capabilities with support for both Turkish and English languages. It includes
advanced text preprocessing, multiple tokenization strategies, and comprehensive
visualization and export features.

Main Features:
- Support for Turkish and English text processing
- Multiple NMF algorithm variants (standard NMF and orthogonal projective NMF)
- Advanced tokenization (BPE, WordPiece for Turkish; traditional for English)
- Comprehensive text preprocessing and cleaning
- Word cloud generation and topic visualization
- Excel export and database storage
- Coherence score calculation for model evaluation

Example Usage:
    >>> from manta import run_topic_analysis
    >>> result = run_topic_analysis(
    ...     "data.csv",
    ...     column="text",
    ...     language="TR",
    ...     topics=5
    ... )
    >>> print(f"Found {len(result['topic_word_scores'])} topics")

Command Line Usage:
    $ manta analyze data.csv --column text --language TR --topics 5 --wordclouds
"""

# Version information
__version__ = "0.9.1"
__author__ = "Emir Kyz"
__email__ = "emirkyzmain@gmail.com"

# Lazy import for EmojiMap to keep it in public API while hiding internal modules
from typing import Any


def __getattr__(name):
    """Lazy import for public API components."""
    if name == "EmojiMap":
        from ._functions.common_language.emoji_processor import EmojiMap
        return EmojiMap
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Public API exports
__all__ = [
    # Main functions
    "run_topic_analysis",
    "run_optimization",
    # Version info
    "__version__",
    "__author__",
    "__email__",
]


def run_topic_analysis(
    filepath: str = None,
    dataframe = None,
    column: str = None,
    separator: str = ",",
    language: str = "EN",
    topic_count: int = 5,
    nmf_method: str = "nmf",
    lemmatize: bool = False,
    tokenizer_type: str = "bpe",
    words_per_topic: int = 15,
    n_grams_to_discover : Any = None,
    word_pairs_out: bool = True,
    generate_wordclouds: bool = True,
    export_excel: bool = True,
    topic_distribution: bool = True,
    filter_app: bool = False,
    data_filter_options: dict = None,
    emoji_map: bool = False,
    output_name: str = None,
    save_to_db: bool = False,
    output_dir: str = None,
    pagerank_column: str = None,
    **kwargs
) -> dict:
    """
    Perform comprehensive topic modeling analysis on text data using Non-negative Matrix Factorization (NMF).

    This high-level API provides an easy-to-use interface for topic modeling with sensible defaults.
    It supports both Turkish and English languages with various preprocessing and output options.
    Data can be provided as either a file path or a pandas DataFrame.

    Parameters:
        filepath: Absolute path to the input file (CSV or Excel format). Optional if dataframe provided.
        dataframe: Pandas DataFrame containing text data. Optional if filepath provided.
        column: Name of the column containing text data to analyze
        separator: CSV file separator (default: ",")
        language: Language of text data - "TR" for Turkish, "EN" for English (default: "EN")
        topic_count: Number of topics to extract. Defaults to 5 for general use. Set to -1 to auto-select the theoretical maximum number of topics.
        words_per_topic: Number of top words to show per topic (default: 15 for general use.) Use 10-20 for most cases.
        nmf_method: NMF algorithm variant - "nmf", "nmtf", or "pnmf". Defaults to "nmf".
        lemmatize: Apply lemmatization for English text (default: False)
        tokenizer_type: Tokenization method for Turkish - "bpe" or "wordpiece" (default: "bpe")
        word_pairs_out: Create word pairs output (default: True)
        n_grams_to_discover: Discover top n n-grams via BPE style algorithm. Set None to not disvoer, Set to k(int) to discover k amount of n-grams (default: None)
        generate_wordclouds: Create word cloud visualizations (default: True)
        export_excel: Export results to Excel format (default: True)
        topic_distribution: Generate topic distribution plots (default: True)
        filter_app: Filter data by application name (default: False)
        data_filter_options: Dictionary containing filter options for data filtering:
            - filter_app_name: Application name to filter by (default: "")
            - filter_app_column: Column name for application filtering
            - filter_app_country: Country code to filter by (default: "")
            - filter_app_country_column: Column name for country filtering
        save_to_db: Whether to persist data to database (default: False)
        emoji_map: Enable emoji processing (default: False)
        output_name: Custom name for output directory (default: auto-generated)
        output_dir: Base directory for outputs. Defaults to current working directory.
        pagerank_column: Column name containing PageRank scores to use for TF-IDF weighting.
            If provided, documents with higher PageRank get boosted TF-IDF scores (range 1-2x).
        **kwargs: Additional parameters to pass through to the analysis pipeline (e.g., visualization options)
    Returns:
        Dict containing:
            - state: "SUCCESS" if completed successfully, "FAILURE" if error occurred
            - message: Descriptive message about the processing outcome
            - data_name: Name of the processed dataset
            - results: List of results for each NMF variant (if multiple)
    Raises:
        ValueError: For invalid language code, unsupported file format, or if both/neither filepath and dataframe provided
        FileNotFoundError: If input file path does not exist
        KeyError: If specified column is missing from input data.
    Example:
        >>> # Basic usage for Turkish text from file
        >>> result = run_topic_analysis(
        ...     filepath="reviews.csv",
        ...     column="review_text",
        ...     language="TR",
        ...     topic_count=5,
        ...     generate_wordclouds=True,
        ...     export_excel=True
        ... )

        >>> # Using DataFrame input
        >>> import pandas as pd
        >>> df = pd.read_csv("reviews.csv")
        >>> result = run_topic_analysis(
        ...     dataframe=df,
        ...     column="review_text",
        ...     language="TR",
        ...     topic_count=5
        ... )

        >>> # Check results
        >>> print(f"Analysis state: {result['state']}")
    :note:
        - Creates output directories for storing results and visualizations
        - Automatically handles file preprocessing and data cleaning
        - Supports both CSV (with automatic delimiter detection) and Excel files
        - DataFrame input allows for custom preprocessing before topic modeling

    """
    from pathlib import Path

    from .config import create_config_from_params

    # Import dependencies only when needed
    from .manta_entry import run_manta_process

    # Validate inputs
    if filepath is None and dataframe is None:
        raise ValueError("Either filepath or dataframe must be provided")
    if filepath is not None and dataframe is not None:
        raise ValueError("Cannot provide both filepath and dataframe - choose one")

    # Create configuration object from function parameters
    config = create_config_from_params(
        language=language,
        topic_count=topic_count,
        nmf_method=nmf_method,
        lemmatize=lemmatize,
        tokenizer_type=tokenizer_type,
        words_per_topic=words_per_topic,
        n_grams_to_discover = n_grams_to_discover,
        word_pairs_out=word_pairs_out,
        generate_wordclouds=generate_wordclouds,
        export_excel=export_excel,
        topic_distribution=topic_distribution,
        separator=separator,
        filter_app=filter_app,
        data_filter_options=data_filter_options,
        emoji_map=emoji_map,
        save_to_db=save_to_db,
        output_name=output_name,
        pagerank_column=pagerank_column,
        **kwargs
    )

    # Set output name if not provided
    if config.output_name is None:
        if filepath:
            config.output_name = config.generate_output_name(filepath)
        else:
            # Generate name for DataFrame input
            config.output_name = f"dataframe_{nmf_method}_{tokenizer_type}_{topic_count}"

    # Convert config to run_options format
    run_options = config.to_run_options()

    # Prepare filepath argument
    resolved_filepath = None
    if filepath:
        resolved_filepath = str(Path(filepath).resolve())

    # Run the analysis
    return run_manta_process(
        filepath=resolved_filepath,
        dataframe=dataframe,
        table_name=run_options['output_name'],
        desired_columns=column,
        options=run_options,
        output_base_dir=output_dir
    )


def run_optimization(
    filepath: str = None,
    dataframe=None,
    column: str = None,
    separator: str = ",",
    language: str = "EN",
    min_topics: int = 2,
    max_topics: int = 20,
    n_grams_to_discover : Any = None,
    step: int = 1,
    nmf_method: str = "nmf",
    lemmatize: bool = False,
    tokenizer_type: str = "bpe",
    words_per_topic: int = 15,
    lambda_val: float = 0.6,
    save_plot: bool = True,
    show_plot: bool = False,
    save_csv: bool = True,
    save_json: bool = True,
    output_dir: str = None,
    pagerank_column: str = None,
    use_cache: bool = True,
    force_reprocess: bool = False,
    **kwargs,
) -> dict:
    """
    Find the optimal number of topics for a dataset using coherence score evaluation.

    This function runs NMF topic modeling with different numbers of topics and
    evaluates each using Gensim C_V coherence score. It returns the optimal
    topic count along with an alternative recommendation based on elbow detection.

    Parameters:
        filepath: Path to input file (CSV or Excel). Optional if dataframe provided.
        dataframe: Pandas DataFrame. Optional if filepath provided.
        column: Name of column containing text to analyze.
        separator: CSV separator (default: ",").
        language: Language code - "TR" or "EN" (default: "EN").
        min_topics: Minimum number of topics to evaluate (default: 2).
        max_topics: Maximum number of topics to evaluate (default: 20).
        step: Step size between topic counts (default: 1).
        nmf_method: NMF algorithm - "nmf", "pnmf", "nmtf" (default: "nmf").
        lemmatize: Apply lemmatization for English (default: False).
        tokenizer_type: Tokenizer for Turkish - "bpe" or "wordpiece" (default: "bpe").
        words_per_topic: Words per topic for coherence calculation (default: 15).
        lambda_val: Lambda value for relevance scoring (default: 0.6).
        save_plot: Save coherence plot to file (default: True).
        show_plot: Display interactive plot (default: False).
        save_csv: Save results to CSV (default: True).
        save_json: Save results to JSON (default: True).
        output_dir: Output directory (default: current directory).
        pagerank_column: Column with PageRank scores for TF-IDF weighting.
        use_cache: Use cached TF-IDF matrix if available (default: True).
        force_reprocess: Force reprocessing, ignore cache (default: False).
        **kwargs: Additional parameters.

    Returns:
        Dict containing:
            - state: "SUCCESS" or "FAILURE"
            - optimal_topic_count: Recommended number of topics (highest coherence)
            - optimal_coherence: Coherence score at optimal
            - elbow_topic_count: Alternative recommendation from elbow detection
            - results: Full results dictionary with all scores
            - output_dir: Path to output files

    Example:
        >>> from manta import run_optimization
        >>> result = run_optimization(
        ...     filepath="papers.csv",
        ...     column="abstract",
        ...     language="EN",
        ...     min_topics=5,
        ...     max_topics=30
        ... )
        >>> print(f"Optimal: {result['optimal_topic_count']} topics")
        >>> print(f"Coherence: {result['optimal_coherence']:.4f}")
        >>> print(f"Elbow point: {result['elbow_topic_count']} topics")

        >>> # Use result in full analysis
        >>> from manta import run_topic_analysis
        >>> analysis = run_topic_analysis(
        ...     filepath="papers.csv",
        ...     column="abstract",
        ...     topic_count=result['optimal_topic_count']
        ... )
    """
    from pathlib import Path

    from .config import DataFilterOptions, OptimizationConfig
    from .optimization_entry import run_optimization_process

    # Validate inputs
    if filepath is None and dataframe is None:
        raise ValueError("Either filepath or dataframe must be provided")
    if filepath is not None and dataframe is not None:
        raise ValueError("Cannot provide both filepath and dataframe - choose one")

    # Create optimization config (kwargs allows passing n_grams_to_discover, etc.)
    config = OptimizationConfig(
        min_topics=min_topics,
        max_topics=max_topics,
        step=step,
        nmf_method=nmf_method,
        words_per_topic=words_per_topic,
        n_grams_to_discover = n_grams_to_discover,
        language=language,
        tokenizer_type=tokenizer_type,
        lemmatize=lemmatize,
        separator=separator,
        lambda_val=lambda_val,
        save_plot=save_plot,
        show_plot=show_plot,
        save_csv=save_csv,
        save_json=save_json,
        pagerank_column=pagerank_column,
        use_cache=use_cache,
        force_reprocess=force_reprocess,
    )

    # Resolve filepath
    resolved_filepath = None
    if filepath:
        resolved_filepath = str(Path(filepath).resolve())

    return run_optimization_process(
        filepath=resolved_filepath,
        dataframe=dataframe,
        column=column,
        config=config,
        output_dir=output_dir,
    )
