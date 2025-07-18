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
__version__ = "0.5.0"
__author__ = "Emir Kyz"
__email__ = "emirkyzmain@gmail.com"

# Lazy import for EmojiMap to keep it in public API while hiding internal modules
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
    "EmojiMap",
    # Version info
    "__version__",
    "__author__",
    "__email__",
]


def run_topic_analysis(
    filepath: str,
    column: str, 
    topic_count: int = 5,
    nmf_method: str = "nmf",
    output_dir: str = None,
    **options
):
    """
    Perform comprehensive topic modeling analysis on text data using Non-negative Matrix Factorization (NMF).

    This high-level API provides an easy-to-use interface for topic modeling with sensible defaults.
    It supports both Turkish and English languages with various preprocessing and output options.

    Args:
        filepath (str): Absolute path to the input file (CSV or Excel format)
        column (str): Name of the column containing text data to analyze
        output_dir (str, optional): Base directory for outputs. Defaults to current working directory.
        topic_count (int): Number of topics to extract (default: 5)
        nmf_method (str): NMF algorithm variant - "nmf" or "opnmf" (default: "nmf")
        **options: Configuration dictionary with the following options:
            Language Options:
                language (str): Language of text data - "TR" for Turkish, "EN" for English (default: "TR")
                lemmatize (bool): Apply lemmatization for English text (default: False)
                tokenizer_type (str): Tokenization method for Turkish - "bpe" or "wordpiece" (default: "bpe")
            
            Topic Modeling Options:
                words_per_topic (int): Number of top words to show per topic (default: 15)
            
            Output Options:
                word_pairs_out (bool): Create word pairs output (default: True)
                generate_wordclouds (bool): Create word cloud visualizations (default: True)
                export_excel (bool): Export results to Excel format (default: True)
                topic_distribution (bool): Generate topic distribution plots (default: True)
                output_name (str): Custom name for output directory (default: auto-generated)
            
            Data Processing Options:
                separator (str): CSV file separator (default: ",")
                filter_app (bool): Filter data by application name (default: False)
                filter_app_name (str): Application name to filter by (default: "")

    Returns:
        dict: Processing results containing:
            - state (str): "SUCCESS" if completed successfully, "FAILURE" if error occurred
            - message (str): Descriptive message about the processing outcome
            - data_name (str): Name of the processed dataset
            - topic_word_scores (dict): Dictionary mapping topic IDs to word-score pairs

    Raises:
        ValueError: If invalid language code or unsupported file format is provided
        FileNotFoundError: If the input file path does not exist
        KeyError: If specified column is missing from the input data

    Example:
        >>> # Basic usage for Turkish text
        >>> result = run_topic_analysis(
        ...     "reviews.csv",
        ...     column="review_text",
        ...     language="TR",
        ...     topic_count=5
        ... )
        >>> 
        >>> # Advanced usage for English text with custom options
        >>> result = run_topic_analysis(
        ...     "articles.xlsx",
        ...     column="content",
        ...     language="EN",
        ...     topic_count=10,
        ...     lemmatize=True,
        ...     generate_wordclouds=True,
        ...     export_excel=True
        ... )
        >>> 
        >>> # Access topic modeling results
        >>> topics = result['topic_word_scores']
        >>> print(f"Analysis status: {result['state']}")

    Note:
        - Creates output directories for storing results and visualizations
        - Automatically handles file preprocessing and data cleaning
        - Supports both CSV (with automatic delimiter detection) and Excel files
    """
    import os
    from pathlib import Path
    
    # Import dependencies only when needed
    from .standalone_nmf import run_standalone_nmf 
    from ._functions.common_language.emoji_processor import EmojiMap
    
    # Set defaults
    options.setdefault('language', 'EN')
    options.setdefault('topic_count', topic_count if topic_count is not None else 5)
    options.setdefault('words_per_topic', 15)
    options.setdefault('nmf_method', nmf_method if nmf_method is not None else 'nmf')
    options.setdefault('tokenizer_type', 'bpe')
    options.setdefault('lemmatize', True)
    options.setdefault('generate_wordclouds', True)
    options.setdefault('export_excel', True)
    options.setdefault('topic_distribution', True)
    options.setdefault('separator', ',')
    options.setdefault('filter_app', False)
    options.setdefault('filter_app_name', '')
    options.setdefault('emoji_map', False)
    options.setdefault('word_pairs_out', False)
    
    try:
        filename = filepath.split("/")[-1].split(".")[0].split("_")[0]
    except:
        filename = Path(filepath).name
        
    # Generate output name if not provided
    if 'output_name' not in options:
        filepath_obj = Path(filepath)
        base_name = filepath_obj.stem
        options['output_name'] = f"{base_name}_{options['nmf_method']}_{options['tokenizer_type']}_{options['topic_count']}"


        
    # Create emoji map

    # Build options dictionary for run_standalone_nmf
    run_options = {
        "LANGUAGE": options['language'].upper(),
        "DESIRED_TOPIC_COUNT": options['topic_count'],
        "N_TOPICS": options['words_per_topic'],
        "LEMMATIZE": options['lemmatize'],
        "tokenizer_type": options['tokenizer_type'],
        "tokenizer": None,
        "nmf_type": options['nmf_method'],
        "separator": options['separator'],
        "word_pairs_out": options['word_pairs_out'],
        "gen_cloud": options['generate_wordclouds'],
        "save_excel": options['export_excel'],
        "gen_topic_distribution": options['topic_distribution'],
        "filter_app": options['filter_app'],
        "filter_app_name": options['filter_app_name'],
        "emoji_map": options["emoji_map"]
    }
    
    #TODO: APP name based options will be implemented.
    
    # Run the analysis
    return run_standalone_nmf(
        filepath=str(Path(filepath).resolve()),
        table_name=options['output_name'],
        desired_columns=column,
        options=run_options,
        output_base_dir=output_dir
    )