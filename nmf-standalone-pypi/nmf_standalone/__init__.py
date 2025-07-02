"""
NMF Standalone - A comprehensive topic modeling library for Turkish and English texts.

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
    >>> from nmf_standalone import run_topic_analysis
    >>> result = run_topic_analysis(
    ...     "data.csv", 
    ...     column="text", 
    ...     language="TR", 
    ...     topics=5
    ... )
    >>> print(f"Found {len(result['topic_word_scores'])} topics")

Command Line Usage:
    $ nmf-standalone analyze data.csv --column text --language TR --topics 5 --wordclouds
"""
from .functions.common_language.emoji_processor import EmojiMap

# Version information
__version__ = "0.2.7"
__author__ = "Emir Kyz"
__email__ = "emirkyzmain@gmail.com"

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
    output_dir: str = None,
    **options
):
    """
    High-level API for topic modeling analysis.
    
    This is a simplified interface to the full NMF standalone functionality,
    providing sensible defaults for common use cases.
    
    Args:
        filepath (str): Path to input CSV or Excel file
        column (str): Name of column containing text data
        **options: Configuration options:
            - language (str): "TR" for Turkish, "EN" for English (default: "TR")
            - topics (int): Number of topics to extract (default: 5)
            - words_per_topic (int): Top words to show per topic (default: 15)
            - nmf_method (str): "nmf" or "opnmf" algorithm variant (default: "nmf")
            - tokenizer_type (str): "bpe" or "wordpiece" for Turkish (default: "bpe")
            - lemmatize (bool): Apply lemmatization for English (default: False)
            - generate_wordclouds (bool): Create word cloud visualizations (default: True)
            - export_excel (bool): Export results to Excel (default: True)
            - topic_distribution (bool): Generate distribution plots (default: True)
            - output_name (str): Custom output directory name (default: auto-generated)
            - output_dir (str): Base directory for outputs (default: current working directory)
            - separator (str): Separator for text processing (default: "|")
            - filter_app (bool): Enable app filtering (default: False)
            - filter_app_name (str): App name for filtering (default: "")
        
    Returns:
        dict: Analysis results with topics and word scores
        
    Example:
        >>> result = run_topic_analysis(
        ...     "reviews.csv",
        ...     column="review_text", 
        ...     language="TR",
        ...     topics=7,
        ...     generate_wordclouds=True
        ... )
        >>> topics = result['topic_word_scores']
    """
    import os
    
    # Import dependencies only when needed
    from .standalone_nmf import run_standalone_nmf 
    from .functions.common_language.emoji_processor import EmojiMap
    
    # Set defaults
    options.setdefault('language', 'EN')
    options.setdefault('topics', 5)
    options.setdefault('words_per_topic', 15)
    options.setdefault('nmf_method', 'nmf')
    options.setdefault('tokenizer_type', 'bpe')
    options.setdefault('lemmatize', True)
    options.setdefault('generate_wordclouds', True)
    options.setdefault('export_excel', True)
    options.setdefault('topic_distribution', True)
    options.setdefault('separator', ',')
    options.setdefault('filter_app', False)
    options.setdefault('filter_app_name', '')
    
    # Generate output name if not provided
    if 'output_name' not in options:
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        options['output_name'] = f"{base_name}_{options['nmf_method']}_{options['tokenizer_type']}_{options['topics']}"
    
    # Create emoji map
    emoji_map = EmojiMap()
    
    # Build options dictionary for run_standalone_nmf
    run_options = {
        "LANGUAGE": options['language'].upper(),
        "DESIRED_TOPIC_COUNT": options['topics'],
        "N_TOPICS": options['words_per_topic'],
        "LEMMATIZE": options['lemmatize'],
        "tokenizer_type": options['tokenizer_type'],
        "tokenizer": None,
        "nmf_type": options['nmf_method'],
        "separator": options['separator'],
        "gen_cloud": options['generate_wordclouds'],
        "save_excel": options['export_excel'],
        "gen_topic_distribution": options['topic_distribution'],
        "filter_app": options['filter_app'],
        "filter_app_name": options['filter_app_name'],
        "emoji_map": emoji_map
    }
    
    # Run the analysis
    return run_standalone_nmf(
        filepath=os.path.abspath(filepath),
        table_name=options['output_name'],
        desired_columns=column,
        options=run_options,
        output_base_dir=output_dir
    )