"""
Text processing pipeline for MANTA topic analysis.
"""

from typing import Dict, Any, Optional, Tuple

import pandas as pd

from .._functions.english.english_entry import process_english_file
from .._functions.turkish.turkish_entry import process_turkish_file
from ..utils.console.console_manager import ConsoleManager, get_console


class TextPipeline:
    """Handles language-specific text processing and feature extraction."""
    
    @staticmethod
    def perform_text_processing(
        df: pd.DataFrame, 
        desired_columns: str, 
        options: Dict[str, Any], 
        console: Optional[ConsoleManager] = None
    ) -> Tuple[Any, Any, Any, Any, Any, Dict[str, Any]]:
        """
        Perform language-specific text processing and feature extraction.

        Args:
            df: Preprocessed DataFrame
            desired_columns: Column containing text data
            options: Configuration options
            console: Console manager for status messages

        Returns:
            Tuple of (tdm, vocab, counterized_data, text_array, original_text_array, updated_options)
        """
        _console = console or get_console()
        _console.print_status(f"Starting text processing ({options['LANGUAGE']})...", "processing")

        # Preserve original text before preprocessing
        original_text_array = df[desired_columns].values.copy()

        # Get PageRank weights from options (may be None)
        pagerank_weights = options.get("pagerank_weights")

        if options["LANGUAGE"] == "TR":
            tdm, vocab, counterized_data, text_array, options["tokenizer"], options["emoji_map"] = (
                process_turkish_file(
                    df,
                    desired_columns,
                    options["tokenizer"],
                    tokenizer_type=options["tokenizer_type"],
                    emoji_map=options["emoji_map"],
                    enable_ngram_bpe=options.get("enable_ngram_bpe", False),
                    ngram_vocab_limit=options.get("ngram_vocab_limit", 10000),
                    min_pair_frequency=options.get("min_pair_frequency", 2),
                    pagerank_weights=pagerank_weights
                )
            )
        elif options["LANGUAGE"] == "EN":
            tdm, vocab, counterized_data, text_array, options["emoji_map"] = process_english_file(
                df,
                desired_columns,
                options["LEMMATIZE"],
                emoji_map=options["emoji_map"],
                n_gram_discover_count=options.get("n_grams_to_discover", None),
                ngram_vocab_limit=options.get("ngram_vocab_limit", 10000),
                min_pair_frequency=options.get("min_pair_frequency", 2),
                pagerank_weights=pagerank_weights,
                keep_numbers=options.get("keep_numbers", False),
                ngram_auto_k=options.get("ngram_auto_k", 0.5),
                filter_standalone_numbers=options.get("filter_standalone_numbers", True),
                use_pmi=options.get("use_pmi", True),
                console=console
            )
        else:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}")

        _console.print_status("Text processing completed", "success")
        del df 
        
        return tdm, vocab, counterized_data, text_array, original_text_array, options
