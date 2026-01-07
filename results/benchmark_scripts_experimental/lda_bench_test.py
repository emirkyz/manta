#!/usr/bin/env python3
"""
LDA Benchmark Test - Equivalent to no_bench_test.py but for LDA

This test runs LDA topic modeling on the same dataset used for NMF testing,
using identical preprocessing parameters for fair comparison.
"""

import os
import sys

# Add parent directory to path to import standalone_lda
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from standalone_lda import compare_lda_with_nmf, StandaloneLDA
from manta._functions.common_language.emoji_processor import EmojiMap

def run_lda_topic_analysis(
    filepath: str,
    column: str,
    separator: str = ",",
    language: str = "EN",
    tokenizer_type: str = "bpe",  # Not used in LDA but kept for compatibility
    lemmatize: bool = True,
    generate_wordclouds: bool = True,
    topic_count: int = 5,
    words_per_topic: int = 15,
    emoji_map: bool = True,
    word_pairs_out: bool = False,  # Not implemented in LDA but kept for compatibility
    lda_method: str = "lda",  # Equivalent to nmf_method
    filter_app: bool = False,  # Not implemented but kept for compatibility
    data_filter_options: dict = None,  # Not implemented but kept for compatibility
    save_to_db: bool = False,  # Not implemented but kept for compatibility
    max_iter: int = 50,  # LDA-specific parameter
    random_state: int = 42  # LDA-specific parameter
) -> dict:
    """
    Run LDA topic analysis with parameters matching the NMF interface.
    
    This function provides the same interface as the NMF topic analysis
    but uses LDA instead of NMF for topic modeling.
    
    Args:
        filepath: Path to the input data file
        column: Column name containing text data
        separator: CSV separator (default: ",")
        language: Language of the text ("EN" or "TR")
        tokenizer_type: Tokenizer type (kept for compatibility, not used in LDA)
        lemmatize: Whether to apply lemmatization
        generate_wordclouds: Whether to generate wordcloud images
        topic_count: Number of topics to discover
        words_per_topic: Number of words to display per topic
        emoji_map: Whether to use emoji mapping
        word_pairs_out: Whether to output word pairs (not implemented in LDA)
        lda_method: LDA method (kept for compatibility)
        filter_app: Whether to filter by app (not implemented)
        data_filter_options: Filter options (not implemented)
        save_to_db: Whether to save to database (not implemented)
        max_iter: Maximum iterations for LDA
        random_state: Random seed for reproducible results
        
    Returns:
        Dictionary with LDA results
    """
    
    print("=== LDA Topic Analysis ===")
    print(f"Dataset: {filepath}")
    print(f"Column: {column}")
    print(f"Language: {language}")
    print(f"Topics: {topic_count}")
    print(f"Words per topic: {words_per_topic}")
    print(f"Lemmatization: {lemmatize}")
    print(f"Wordclouds: {generate_wordclouds}")
    
    # Create emoji map if requested
    emoji_processor = EmojiMap() if emoji_map else None
    
    # Extract base name from filepath for output naming
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    base_name = f"{base_name}_lda_{topic_count}topics"
    
    try:
        # Run LDA analysis using the standalone LDA implementation
        results = compare_lda_with_nmf(
            filepath=filepath,
            desired_columns=column,
            num_topics=topic_count,
            base_name=base_name,
            lemmatize=lemmatize,
            num_words=words_per_topic,
            generate_wordclouds=generate_wordclouds,
            separator=separator
        )
        
        print(f"\n✅ LDA Analysis Complete!")
        print(f"Coherence Score: {results['coherence_score']:.4f}")
        print(f"Output saved to: {results['output_dir']}")
        
        # Print topics for immediate review
        print(f"\n📊 Discovered Topics:")
        for i, (topic_name, words) in enumerate(results['topics_data'].items(), 1):
            word_list = [word.split(':')[0] for word in words[:10]]  # Show top 10 words
            print(f"{i:2d}. {topic_name}: {', '.join(word_list)}")
        
        return {
            "status": "SUCCESS",
            "coherence_score": results['coherence_score'],
            "topics_data": results['topics_data'],
            "output_directory": results['output_dir'],
            "topic_count": topic_count,
            "method": "LDA"
        }
        
    except Exception as e:
        print(f"❌ Error in LDA analysis: {str(e)}")
        return {
            "status": "FAILURE", 
            "error": str(e),
            "method": "LDA"
        }

if __name__ == "__main__":
    # Test parameters (matching no_bench_test.py exactly)
    file_path = "../../datasets/bbc_news.csv"
    column = "text"
    
    # Run LDA with same parameters as NMF test
    result = run_lda_topic_analysis(
        filepath=file_path,
        column=column,
        separator=",",
        language="EN",
        tokenizer_type="bpe",  # Not used in LDA but kept for compatibility
        lemmatize=True,
        generate_wordclouds=True,
        topic_count=5,
        words_per_topic=15,
        emoji_map=True,
        word_pairs_out=False,
        lda_method="lda",  # Equivalent to nmf_method
        filter_app=False,
        data_filter_options={
            "filter_app_country": "TR",
            "filter_app_country_column": "REVIEWER_LANGUAGE",
        },
        save_to_db=False,
        max_iter=50,  # LDA-specific parameter
        random_state=42  # For reproducible results
    )
    
    # Print final results
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Status: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"Method: {result['method']}")
        print(f"Topics: {result['topic_count']}")
        print(f"Coherence: {result['coherence_score']:.4f}")
        print(f"Output: {result['output_directory']}")
    else:
        print(f"Error: {result['error']}")