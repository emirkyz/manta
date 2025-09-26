"""
Text processing pipeline for MANTA topic analysis.
"""

from typing import Dict, Any, Optional, Tuple

import pandas as pd

from .._functions.english.english_entry import process_english_file
from .._functions.turkish.turkish_entry import process_turkish_file
from ..utils.console.console_manager import ConsoleManager


class TextPipeline:
    """Handles language-specific text processing and feature extraction."""
    
    @staticmethod
    def perform_text_processing(
        df: pd.DataFrame, 
        desired_columns: str, 
        options: Dict[str, Any], 
        console: Optional[ConsoleManager] = None
    ) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        """
        Perform language-specific text processing and feature extraction.
        
        Args:
            df: Preprocessed DataFrame
            desired_columns: Column containing text data
            options: Configuration options
            console: Console manager for status messages
            
        Returns:
            Tuple of (tdm, vocab, counterized_data, text_array, updated_options)
        """
        if console:
            console.print_status(f"Starting text processing ({options['LANGUAGE']})...", "processing")
        else:
            print("Starting preprocessing...")
        
        if options["LANGUAGE"] == "TR":
            tdm, vocab, counterized_data, text_array, options["tokenizer"], options["emoji_map"] = (
                process_turkish_file(
                    df,
                    desired_columns,
                    options["tokenizer"],
                    tokenizer_type=options["tokenizer_type"],
                    emoji_map=options["emoji_map"],
                )
            )
        elif options["LANGUAGE"] == "EN":
            tdm, vocab, counterized_data, text_array, options["emoji_map"] = process_english_file(
                df,
                desired_columns,
                options["LEMMATIZE"],
                emoji_map=options["emoji_map"],
            )
        else:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}")
        
        if console:
            console.print_status("Text processing completed", "success")
        
        return tdm, vocab, counterized_data, text_array, options
