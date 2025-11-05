"""
Configuration classes and validation for MANTA topic analysis.

This module contains the configuration dataclasses and validation logic
that were previously embedded in __init__.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


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
    save_to_db: bool = False
    data_filter_options: DataFilterOptions = field(default_factory=DataFilterOptions)
    output_name: Optional[str] = None
    enable_ngram_bpe: bool = False
    ngram_vocab_limit: int = 10000
    min_pair_frequency: int = 2
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
            "min_pair_frequency": self.min_pair_frequency
        }
        
        # Merge additional parameters from kwargs
        # Additional params take precedence over defaults but not over explicitly set config values
        for key, value in self.additional_params.items():
            if key not in options:  # Only add if not already present
                options[key] = value
        
        return options


def create_config_from_params(
    language: str = "EN",
    topic_count: int = 5,
    nmf_method: str = "nmf",
    lemmatize: bool = False,
    tokenizer_type: str = "bpe",
    words_per_topic: int = 15,
    word_pairs_out: bool = True,
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
        save_to_db=save_to_db,
        data_filter_options=dfo,
        output_name=output_name,
        enable_ngram_bpe=enable_ngram_bpe,
        ngram_vocab_limit=ngram_vocab_limit,
        min_pair_frequency=min_pair_frequency,
        additional_params=kwargs
    )
