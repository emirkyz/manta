#!/usr/bin/env python3
"""
Command-line interface for MANTA (Multi-lingual Advanced NMF-based Topic Analysis).

This module provides a command-line interface for the MANTA topic modeling
functionality, allowing users to analyze text data from CSV or Excel files
and extract topics using Non-negative Matrix Factorization.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any

from .standalone_nmf import run_standalone_nmf
from ._functions.common_language.emoji_processor import EmojiMap


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="MANTA - Multi-lingual Advanced NMF-based Topic Analysis tool for Turkish and English texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Turkish app reviews with 5 topics
  manta analyze reviews.csv --column REVIEW --language TR --topics 5
  
  # Analyze English documents with lemmatization and word clouds
  manta analyze docs.xlsx --column text --language EN --topics 10 --lemmatize --wordclouds
  
  # Use BPE tokenizer for Turkish text
  manta analyze data.csv --column content --language TR --tokenizer bpe --topics 7
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze', 
        help='Perform topic modeling analysis on text data'
    )
    
    # Required arguments
    analyze_parser.add_argument(
        'filepath',
        help='Path to input file (CSV or Excel format)'
    )
    
    analyze_parser.add_argument(
        '--column', '-c',
        required=True,
        help='Name of the column containing text data to analyze'
    )
    
    analyze_parser.add_argument(
        '--language', '-l',
        choices=['TR', 'EN'],
        required=True,
        help='Language of the text data (TR for Turkish, EN for English)'
    )
    
    analyze_parser.add_argument(
        '--topics', '-t',
        type=int,
        default=5,
        help='Number of topics to extract (default: 5)'
    )
    
    # Optional arguments
    analyze_parser.add_argument(
        '--output-name', '-o',
        help='Custom name for output files (default: derived from input filename)'
    )
    
    analyze_parser.add_argument(
        '--output-dir',
        help='Directory to save output files (default: current working directory)'
    )
    
    analyze_parser.add_argument(
        '--tokenizer',
        choices=['bpe', 'wordpiece'],
        default='bpe',
        help='Tokenizer type for Turkish text (default: bpe)'
    )
    
    analyze_parser.add_argument(
        '--nmf-method',
        choices=['nmf', 'opnmf'],
        default='nmf',
        help='NMF algorithm variant (default: nmf)'
    )
    
    analyze_parser.add_argument(
        '--words-per-topic',
        type=int,
        default=15,
        help='Number of top words to display per topic (default: 15)'
    )
    
    analyze_parser.add_argument(
        '--lemmatize',
        action='store_true',
        help='Apply lemmatization for English text preprocessing'
    )
    
    analyze_parser.add_argument(
        '--wordclouds',
        action='store_true',
        help='Generate word cloud visualizations for topics'
    )
    
    analyze_parser.add_argument(
        '--excel',
        action='store_true',
        help='Export results to Excel format'
    )
    
    analyze_parser.add_argument(
        '--topic-distribution',
        action='store_true',
        help='Generate topic distribution plots'
    )
    
    analyze_parser.add_argument(
        '--separator',
        default=',',
        help='CSV separator character (default: |)'
    )
    
    analyze_parser.add_argument(
        '--filter-app',
        help='Filter data by specific app name (for app review datasets)'
    )
    
    analyze_parser.add_argument(
        '--word-pairs',
        action='store_true',
        help='Generate word co-occurrence analysis and heatmap'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    filepath = Path(args.filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {args.filepath}")
    
    if not filepath.suffix.lower() in {'.csv', '.xlsx', '.xls'}:
        raise ValueError("Input file must be in CSV or Excel format")
    
    if args.topics < 1:
        raise ValueError("Number of topics must be at least 1")
    
    if args.words_per_topic < 1:
        raise ValueError("Words per topic must be at least 1")


def build_options(args: argparse.Namespace) -> Dict[str, Any]:
    """Build options dictionary from command-line arguments."""
    # Create emoji map for Turkish processing
    emoji_map = EmojiMap()
    
    # Generate output name if not provided
    if args.output_name:
        table_name = args.output_name
    else:
        filepath = Path(args.filepath)
        base_name = filepath.stem
        table_name = f"{base_name}_{args.nmf_method}_{args.tokenizer}_{args.topics}"
    
    options = {
        "LANGUAGE": args.language,
        "DESIRED_TOPIC_COUNT": args.topics,
        "N_TOPICS": args.words_per_topic,
        "LEMMATIZE": args.lemmatize,
        "tokenizer_type": args.tokenizer,
        "tokenizer": None,  # Will be initialized in run_standalone_nmf
        "nmf_type": args.nmf_method,
        "separator": args.separator,
        "gen_cloud": args.wordclouds,
        "save_excel": args.excel,
        "gen_topic_distribution": args.topic_distribution,
        "filter_app": bool(args.filter_app),
        "filter_app_name": args.filter_app or "",
        "emoji_map": emoji_map,
        "word_pairs_out": args.word_pairs
    }
    
    return options, table_name


def analyze_command(args: argparse.Namespace) -> int:
    """Execute the analyze command."""
    try:
        # Validate arguments
        validate_arguments(args)
        
        # Build options
        options, table_name = build_options(args)
        
        # Convert to absolute path
        filepath = Path(args.filepath).resolve()
        
        print(f"Starting MANTA topic modeling analysis...")
        print(f"Input file: {filepath}")
        print(f"Language: {args.language}")
        print(f"Topics: {args.topics}")
        print(f"Column: {args.column}")
        print(f"Output name: {table_name}")
        print("-" * 50)
        
        # Run the analysis
        result = run_standalone_nmf(
            filepath=filepath,
            table_name=table_name,
            desired_columns=args.column,
            options=options,
            output_base_dir=args.output_dir
        )
        
        if result["state"] == "SUCCESS":
            print("\n" + "=" * 50)
            print("✅ Analysis completed successfully!")
            print(f"📊 Dataset: {result['data_name']}")
            print(f"📁 Results saved in: Output/{table_name}/")
            
            if args.wordclouds:
                print("🎨 Word clouds generated")
            if args.excel:
                print("📈 Excel report exported")
            if args.topic_distribution:
                print("📊 Topic distribution plots created")
                
            return 0
        else:
            print(f"\n❌ Analysis failed: {result['message']}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        return analyze_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())