"""
Configuration classes and validation for MANTA topic analysis.

This module contains the configuration dataclasses and validation logic
that were previously embedded in __init__.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataFilterOptions:
    """Options for filtering input data by application or country."""
    filter_app_country: str = ''
    filter_app_country_column: str = ''
    filter_app_name: str = ''
    filter_app_column: str = ''


@dataclass
class TopicAnalysisConfig:
    """Configuration for topic analysis with validation."""

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
    n_grams_to_discover: Any = None  # Can be int, "auto", or None
    ngram_auto_k: float = 0.5  # Scaling constant for auto n-gram formula
    keep_numbers: bool = False  # Preserve numbers for BPE merging
    filter_standalone_numbers: bool = True  # Filter unmerged numbers after BPE
    use_pmi: bool = True  # Use PMI scoring for BPE when keep_numbers is True
    save_to_db: bool = False
    data_filter_options: DataFilterOptions = field(default_factory=DataFilterOptions)
    output_name: Optional[str] = None
    enable_ngram_bpe: bool = False
    ngram_vocab_limit: int = 10000
    min_pair_frequency: int = 2

    # Cache and processing options
    use_cache: bool = True  # Check for cached data (will prompt user interactively if found)
    force_reprocess: bool = False  # Force reprocessing without prompting (skips cache entirely)
    nmf_variants: Optional[List[str]] = None  # List of NMF variants to run (defaults to [nmf_method])
    datetime_column: Optional[str] = None  # Column name for temporal analysis
    pagerank_column: Optional[str] = None  # Column name for PageRank weights (boosts TF-IDF)

    additional_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate all configuration options."""
        # Validate language
        if self.language.upper() not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Must be one of {self.SUPPORTED_LANGUAGES}")

        # Validate topic count
        if self.topic_count <= 0 and self.topic_count != -1:
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

        # Validate n-gram BPE parameters
        if self.ngram_vocab_limit <= 0:
            raise ValueError(f"Invalid ngram_vocab_limit: {self.ngram_vocab_limit}. Must be positive")

        if self.min_pair_frequency <= 0:
            raise ValueError(f"Invalid min_pair_frequency: {self.min_pair_frequency}. Must be positive")

        # Validate NMF variants if provided
        if self.nmf_variants is not None:
            if not isinstance(self.nmf_variants, list):
                raise ValueError("nmf_variants must be a list")
            for variant in self.nmf_variants:
                if variant.lower() not in self.SUPPORTED_NMF_METHODS:
                    raise ValueError(f"Unsupported NMF variant: {variant}. Must be one of {self.SUPPORTED_NMF_METHODS}")

        # Validate cache options consistency
        if self.use_cache and self.force_reprocess:
            raise ValueError("Cannot set both use_cache=True and force_reprocess=True")

    def generate_output_name(self, filepath: str) -> str:
        """Generate a descriptive output name based on input file and configuration."""
        filepath_obj = Path(filepath)
        base_name = filepath_obj.stem
        if self.topic_count <= 0:
            return f"{base_name}_{self.nmf_method}_{self.tokenizer_type}_auto"
        return f"{base_name}_{self.nmf_method}_{self.tokenizer_type}_{self.topic_count}"

    def to_run_options(self) -> Dict:
        """Convert config to format expected by run_standalone_nmf."""
        options = {
            "LANGUAGE": self.language.upper(),
            "DESIRED_TOPIC_COUNT": self.topic_count if self.topic_count is not None else self.topics,
            "N_TOPICS": self.words_per_topic,
            "LEMMATIZE": self.lemmatize,
            "tokenizer_type": self.tokenizer_type,
            "tokenizer": None,
            "nmf_type": self.nmf_method,
            "separator": self.separator,
            "word_pairs_out": self.word_pairs_out,
            "n_grams_to_discover": self.n_grams_to_discover,
            "ngram_auto_k": self.ngram_auto_k,
            "keep_numbers": self.keep_numbers,
            "filter_standalone_numbers": self.filter_standalone_numbers,
            "use_pmi": self.use_pmi,
            "gen_cloud": self.generate_wordclouds,
            "save_excel": self.export_excel,
            "gen_topic_distribution": self.topic_distribution,
            "filter_app": self.filter_app,
            "emoji_map": self.emoji_map,
            "save_to_db": self.save_to_db,
            "data_filter_options": self.data_filter_options.__dict__,
            "output_name": self.output_name,
            "enable_ngram_bpe": self.enable_ngram_bpe,
            "ngram_vocab_limit": self.ngram_vocab_limit,
            "min_pair_frequency": self.min_pair_frequency,
            # New cache and processing options
            "use_cache": self.use_cache,
            "force_reprocess": self.force_reprocess,
            "nmf_variants": self.nmf_variants if self.nmf_variants is not None else [self.nmf_method],
            "datetime_column": self.datetime_column,
            "pagerank_column": self.pagerank_column,
        }

        # Merge additional parameters from kwargs
        # Additional params take precedence over defaults but not over explicitly set config values
        for key, value in self.additional_params.items():
            if key not in options:  # Only add if not already present
                options[key] = value

        return options


@dataclass
class OptimizationConfig:
    """Configuration for topic count optimization.

    This configuration is used for finding the optimal number of topics
    by evaluating coherence scores across a range of topic counts.
    """

    # Supported values (shared with TopicAnalysisConfig)
    SUPPORTED_LANGUAGES = {'EN', 'TR'}
    SUPPORTED_NMF_METHODS = {'nmf', 'nmtf', 'pnmf'}
    SUPPORTED_TOKENIZER_TYPES = {'bpe', 'wordpiece'}

    # Topic range settings
    min_topics: int = 2
    max_topics: int = 20
    step: int = 1

    # NMF settings
    nmf_method: str = 'nmf'
    words_per_topic: int = 15

    # Text processing settings
    language: str = 'EN'
    tokenizer_type: str = 'bpe'
    lemmatize: bool = True
    separator: str = ','
    n_grams_to_discover: Any = None  # int, "auto", or None to disable

    # Coherence settings
    lambda_val: float = 0.6  # For relevance scoring in coherence calculation

    # Data filtering options
    filter_app: bool = False
    data_filter_options: DataFilterOptions = field(default_factory=DataFilterOptions)
    pagerank_column: Optional[str] = None

    # Output settings
    save_plot: bool = True
    save_csv: bool = True
    save_json: bool = True
    show_plot: bool = False

    # Cache settings
    use_cache: bool = True
    force_reprocess: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate all optimization configuration options."""
        # Validate topic range
        if self.min_topics < 1:
            raise ValueError(f"min_topics must be >= 1, got {self.min_topics}")
        if self.max_topics < self.min_topics:
            raise ValueError(
                f"max_topics ({self.max_topics}) must be >= min_topics ({self.min_topics})"
            )
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")

        # Validate NMF method
        if self.nmf_method.lower() not in self.SUPPORTED_NMF_METHODS:
            raise ValueError(
                f"Unsupported NMF method: {self.nmf_method}. "
                f"Must be one of {self.SUPPORTED_NMF_METHODS}"
            )

        # Validate language
        if self.language.upper() not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {self.language}. "
                f"Must be one of {self.SUPPORTED_LANGUAGES}"
            )

        # Validate tokenizer type
        if self.tokenizer_type.lower() not in self.SUPPORTED_TOKENIZER_TYPES:
            raise ValueError(
                f"Unsupported tokenizer type: {self.tokenizer_type}. "
                f"Must be one of {self.SUPPORTED_TOKENIZER_TYPES}"
            )

        # Validate words per topic
        if self.words_per_topic <= 0:
            raise ValueError(f"words_per_topic must be > 0, got {self.words_per_topic}")

        # Validate lambda value
        if not 0 <= self.lambda_val <= 1:
            raise ValueError(f"lambda_val must be between 0 and 1, got {self.lambda_val}")

        # Validate separator
        if not self.separator:
            raise ValueError("separator cannot be empty")

    def get_topic_counts(self) -> List[int]:
        """Return list of topic counts to evaluate."""
        return list(range(self.min_topics, self.max_topics + 1, self.step))

    def to_text_processing_options(self) -> Dict[str, Any]:
        """Convert to options dict for TextPipeline compatibility."""
        return {
            "LANGUAGE": self.language.upper(),
            "LEMMATIZE": self.lemmatize,
            "tokenizer_type": self.tokenizer_type,
            "tokenizer": None,
            "separator": self.separator,
            "N_TOPICS": self.words_per_topic,
            "DESIRED_TOPIC_COUNT": self.min_topics,  # Placeholder, overridden during iteration
            "nmf_type": self.nmf_method,
            "n_grams_to_discover": self.n_grams_to_discover,
            "filter_app": self.filter_app,
            "data_filter_options": (
                self.data_filter_options.__dict__
                if hasattr(self.data_filter_options, '__dict__')
                else self.data_filter_options
            ),
            "pagerank_column": self.pagerank_column,
            "emoji_map": None,
            "gen_cloud": False,  # Disable visualizations for optimization
            "save_excel": False,
            "word_pairs_out": False,
            "gen_topic_distribution": False,
            "save_to_db": False,
            "use_cache": self.use_cache,
            "force_reprocess": self.force_reprocess,
        }


def create_config_from_params(
    language: str = "EN",
    topic_count: int = 5,
    nmf_method: str = "nmf",
    lemmatize: bool = False,
    tokenizer_type: str = "bpe",
    words_per_topic: int = 15,
    word_pairs_out: bool = True,
    n_grams_to_discover: Any = None,
    ngram_auto_k: float = 0.5,
    keep_numbers: bool = False,
    filter_standalone_numbers: bool = True,
    use_pmi: bool = True,
    generate_wordclouds: bool = True,
    export_excel: bool = True,
    topic_distribution: bool = True,
    separator: str = ",",
    filter_app: bool = False,
    data_filter_options: dict = None,
    emoji_map: bool = False,
    save_to_db: bool = False,
    output_name: str = None,
    enable_ngram_bpe: bool = False,
    ngram_vocab_limit: int = 10000,
    min_pair_frequency: int = 2,
    use_cache: bool = True,
    force_reprocess: bool = False,
    nmf_variants: Optional[List[str]] = None,
    datetime_column: Optional[str] = None,
    pagerank_column: Optional[str] = None,
    **kwargs
) -> TopicAnalysisConfig:
    """Create a TopicAnalysisConfig from individual parameters."""
    if data_filter_options is not None:
        dfo = DataFilterOptions(**data_filter_options)
    else:
        dfo = DataFilterOptions()

    return TopicAnalysisConfig(
        language=language,
        topic_count=topic_count,
        words_per_topic=words_per_topic,
        nmf_method=nmf_method,
        tokenizer_type=tokenizer_type,
        lemmatize=lemmatize,
        generate_wordclouds=generate_wordclouds,
        export_excel=export_excel,
        topic_distribution=topic_distribution,
        separator=separator,
        filter_app=filter_app,
        emoji_map=emoji_map,
        word_pairs_out=word_pairs_out,
        n_grams_to_discover=n_grams_to_discover,
        ngram_auto_k=ngram_auto_k,
        keep_numbers=keep_numbers,
        filter_standalone_numbers=filter_standalone_numbers,
        use_pmi=use_pmi,
        save_to_db=save_to_db,
        data_filter_options=dfo,
        output_name=output_name,
        enable_ngram_bpe=enable_ngram_bpe,
        ngram_vocab_limit=ngram_vocab_limit,
        min_pair_frequency=min_pair_frequency,
        use_cache=use_cache,
        force_reprocess=force_reprocess,
        nmf_variants=nmf_variants,
        datetime_column=datetime_column,
        pagerank_column=pagerank_column,
        additional_params=kwargs
    )
