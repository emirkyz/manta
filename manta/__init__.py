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
__version__ = "0.5.5"
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
        topic_count (topics) (int, optional): Number of topics to extract. Defaults to 5.
        nmf_method (str, optional): NMF algorithm variant - "nmf", "nmtf", or "pnmf". Defaults to "nmf".
        output_dir (str, optional): Base directory for outputs. Defaults to current working directory.
        **options: Additional configuration options:

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
                emoji_map (bool): Enable emoji processing (default: False)
    
    Returns:
        dict: Processing results containing:
            state (str): "SUCCESS" if completed successfully, "FAILURE" if error occurred
            message (str): Descriptive message about the processing outcome 
            data_name (str): Name of the processed dataset
            topic_word_scores (dict): Dictionary mapping topic IDs to word-score pairs
    
    Raises:
        ValueError: For invalid language code or unsupported file format
        FileNotFoundError: If input file path does not exist
        KeyError: If specified column is missing from input data.

    Example:
        >>> # Basic usage for Turkish text
        >>> result = run_topic_analysis(
        ...     "reviews.csv",
        ...     column="review_text",
        ...     language="TR",
        ...     topic_count=5,
        ...     generate_wordclouds=True,
        ...     export_excel=True
        ... )
        >>>
        >>> # Check results
        >>> print(f"Found {len(result['topic_word_scores'])} topics")
        >>> # Access topic modeling results
        >>> topics = result['topic_word_scores']
        >>> print(f"Analysis status: {result['state']}")

    Note:
        - Creates output directories for storing results and visualizations
        - Automatically handles file preprocessing and data cleaning
        - Supports both CSV (with automatic delimiter detection) and Excel files
    """
    from pathlib import Path
    
    # Import dependencies only when needed
    from .manta_entry import run_manta_process
    from ._functions.common_language.emoji_processor import EmojiMap
    
    from dataclasses import dataclass, field
    from typing import Dict, Optional

    @dataclass
    class DataFilterOptions:
        filter_app_country: str = ''
        filter_app_country_column: str = ''
        filter_app_name: str = ''
        filter_app_column: str = ''

    @dataclass
    class TopicAnalysisConfig:
        # Supported values for configuration options
        SUPPORTED_LANGUAGES = {'EN', 'TR'}
        SUPPORTED_NMF_METHODS = {'nmf', 'nmtf', 'pnmf'}
        SUPPORTED_TOKENIZER_TYPES = {'bpe', 'wordpiece'}
        
        language: str = 'EN'
        topics: Optional[int] = field(default=5)
        topic_count: int = field(default=5)
        words_per_topic: int = 15
        nmf_method: str = 'nmf'
        tokenizer_type: str = 'bpe'
        lemmatize: bool = True
        generate_wordclouds: bool = True
        export_excel: bool = True
        topic_distribution: bool = True
        separator: str = ','
        filter_app: bool = False
        emoji_map: bool = False
        word_pairs_out: bool = False
        save_to_db: bool = False  # Added save_to_db option
        data_filter_options: DataFilterOptions = field(default_factory=DataFilterOptions)
        output_name: Optional[str] = None

        def __post_init__(self):
            """Validate configuration after initialization."""
            self.validate()
            
        def validate(self) -> None:
            """Validate all configuration options."""
            # Validate language
            if self.language.upper() not in self.SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {self.language}. Must be one of {self.SUPPORTED_LANGUAGES}")
            
            # Validate topic count
            if self.topic_count <= 0:
                raise ValueError(f"Invalid topic_count: {self.topic_count}. Must be positive")
            
            # Validate words per topic
            if self.words_per_topic <= 0:
                raise ValueError(f"Invalid words_per_topic: {self.words_per_topic}. Must be positive")
                
            # Validate NMF method
            if self.nmf_method.lower() not in self.SUPPORTED_NMF_METHODS:
                raise ValueError(f"Unsupported NMF method: {self.nmf_method}. Must be one of {self.SUPPORTED_NMF_METHODS}")
                
            # Validate tokenizer type
            if self.tokenizer_type.lower() not in self.SUPPORTED_TOKENIZER_TYPES:
                raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}. Must be one of {self.SUPPORTED_TOKENIZER_TYPES}")
                
            # Validate separator
            if not self.separator:
                raise ValueError("Separator cannot be empty")
                
            # Validate output_name if provided
            if self.output_name is not None:
                if not isinstance(self.output_name, str):
                    raise ValueError("output_name must be a string")
                if not self.output_name.strip():
                    raise ValueError("output_name cannot be empty or whitespace only")

        @classmethod
        def from_dict(cls, options: Dict, topic_count: int = None, nmf_method: str = None) -> 'TopicAnalysisConfig':
            """Create a TopicAnalysisConfig instance from a dictionary of options."""
            # Handle data filter options
            data_filter_opts = options.get('data_filter_options', {})
            data_filter_config = DataFilterOptions(**data_filter_opts)
            
            # Remove data_filter_options from options to avoid duplicate processing
            options_copy = options.copy()
            options_copy.pop('data_filter_options', None)
            
            # Create config with defaults and override with provided options
            config = cls(
                topic_count=topic_count if topic_count is not None else 5,
                nmf_method=nmf_method if nmf_method is not None else 'nmf',
                data_filter_options=data_filter_config,
                **options_copy
            )
            return config

        def generate_output_name(self, filepath: str) -> str:
            """Generate a descriptive output name based on input file and configuration."""
            filepath_obj = Path(filepath)
            base_name = filepath_obj.stem
            return f"{base_name}_{self.nmf_method}_{self.tokenizer_type}_{self.topic_count}"

        def to_run_options(self) -> Dict:
            """Convert config to format expected by run_standalone_nmf."""
            return {
                "LANGUAGE": self.language.upper(),
                "DESIRED_TOPIC_COUNT": self.topic_count if self.topic_count is not None else self.topics,
                "N_TOPICS": self.words_per_topic,
                "LEMMATIZE": self.lemmatize,
                "tokenizer_type": self.tokenizer_type,
                "tokenizer": None,
                "nmf_type": self.nmf_method,
                "separator": self.separator,
                "word_pairs_out": self.word_pairs_out,
                "gen_cloud": self.generate_wordclouds,
                "save_excel": self.export_excel,
                "gen_topic_distribution": self.topic_distribution,
                "filter_app": self.filter_app,
                "emoji_map": self.emoji_map,
                "save_to_db": self.save_to_db,  # Added save_to_db option
                "data_filter_options": self.data_filter_options.__dict__,
                "output_name": self.output_name  # Added output_name
            }

    # Create configuration object
    config = TopicAnalysisConfig.from_dict(options, topic_count, nmf_method)

    # Set output name if not provided
    if config.output_name is None:
        config.output_name = config.generate_output_name(filepath)

    # Convert config to run_options format
    run_options = config.to_run_options()
    
    #TODO: APP name based options will be implemented.
    
    # Run the analysis
    return run_manta_process(
        filepath=str(Path(filepath).resolve()),
        table_name=run_options['output_name'],
        desired_columns=column,
        options=run_options,
        output_base_dir=output_dir
    )