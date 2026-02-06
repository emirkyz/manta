# MANTA (Multi-lingual Advanced NMF-based Topic Analysis)

[![PyPI version](https://badge.fury.io/py/manta-topic-modelling.svg)](https://badge.fury.io/py/manta-topic-modelling)
[![PyPI version](https://img.shields.io/pypi/v/manta-topic-modelling)](https://badge.fury.io/py/manta-topic-modelling)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive topic modeling system using Non-negative Matrix Factorization (NMF) and Non-negative Matrix Tri-Factorization (NMTF) that supports both English and Turkish text processing. Features advanced tokenization techniques, multiple factorization algorithms including NMTF for topic relationship analysis, and rich visualization capabilities.





### To cite this work;


```bibtex


@article{KARAYAGIZ2025102386,
title = {Manta: Multi-lingual advanced NMF-based topic analysis},
journal = {SoftwareX},
volume = {32},
pages = {102386},
year = {2025},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2025.102386},
url = {https://www.sciencedirect.com/science/article/pii/S2352711025003528},
author = {Emir Karayağız and Tolga Berber},
keywords = {Topic modeling, Non-negative matrix factorization, Python, Natural language processing, Information retrieval},
abstract = {This paper presents MANTA (Multi-lingual Advanced NMF-based Topic Analysis), a novel open-source Python library that provides an integrated pipeline to address key limitations in existing topic modeling workflows. MANTA provides an integrated, easy-to-use pipeline for Non-negative Matrix Factorization (NMF) based topic analysis, uniquely combining corpus-specific subword tokenization (BPE/WordPiece) with advanced term weighting schemes (SMART, BM25) and flexible NMF solver options, including a high-performance Projective NMF method. It offers native support for both English and morphologically complex languages like Turkish. With a simple one-function interface and a command-line utility, MANTA lowers the technical barrier for sophisticated topic analysis, making it a powerful tool for researchers in computational social science and digital humanities.}
}
```


## Quick Start
### Installing locally for Development
To build and run the app locally for development:
First clone the repository:
```bash
git clone https://github.com/emirkyz/manta.git
```
After cloning, navigate to the project directory and create a virtual environment:

```bash
cd manta
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
Next, install the required dependencies. If you have `pip` installed, you can run:
```bash
pip install -e .
```
or if you have `uv` installed, you can use:
```bash
uv pip install -e .
```
### Installation from PyPI
```bash
pip install manta-topic-modelling
```

After that you can import and use the app.

### Python API Usage
```python
from manta import run_topic_analysis

# Simple topic modeling
results = run_topic_analysis(
    filepath="data.csv",
    column="review_text",
    language="EN",
    topic_count=5,
    lemmatize=True
)

# Turkish text analysis
results = run_topic_analysis(
    filepath="turkish_reviews.csv",
    column="yorum_metni",
    language="TR",
    topic_count=8,
    tokenizer_type="bpe",
    generate_wordclouds=True
)

# Using a DataFrame directly instead of filepath
import pandas as pd
df = pd.read_csv("reviews.csv")
results = run_topic_analysis(
    dataframe=df,
    column="review_text",
    language="EN",
    topic_count=5
)

# NMTF analysis for topic relationship discovery
results = run_topic_analysis(
    filepath="data.csv",
    column="text_content",
    language="TR",
    topic_count=6,
    nmf_method="nmtf",
    generate_wordclouds=True
)

# Advanced visualization with t-SNE and LDAvis
results = run_topic_analysis(
    filepath="research_papers.csv",
    column="abstract",
    language="EN",
    topic_count=10,
    gen_tsne=True,                    # Generate t-SNE 2D visualization
    gen_ldavis_plot=True,             # Generate interactive LDAvis
    generate_wordclouds=True
)

# PageRank-weighted TF-IDF for citation-aware topic modeling
results = run_topic_analysis(
    filepath="papers_with_pagerank.csv",
    column="abstract",
    language="EN",
    topic_count=8,
    pagerank_column="pagerank_score"  # Boosts high-PageRank docs 1-2x
)
```

## Result Structure
```
{
"state": State of the analysis, either "success" or "error",
"message": Message about the result of the analysis,
"data_name": Name of the input data file,
"topic_word_scores": JSON object containing topics and their top words with scores,
"topic_doc_scores": JSON object containing topics and their top documents with scores,
"coherence_scores": JSON object containing coherence scores for each topic,
"topic_dist_img": Matplotlib plt object of topic distribution plot if `gen_topic_distribution` is True,
"topic_document_counts": Count of documents per topic,
"topic_relationships": Topic-to-topic relationship matrix (only for NMTF method),
}
```
```
For example:
{
  "state": "success",
  "message": "Analysis completed successfully",
  "data_name": "reviews.csv",
  "topic_word_scores": {
    "topic_0": {
        "word1": 0.15,
        "word2": 0.12,
        "word3": 0.10
        }
    },
  "topic_doc_scores":{
          "topic_0": [
                {
                    "document": "Sample document text...",
                    "score": 0.78
                }
            ],
    }
  "coherence_scores": {
        "gensim": {
           "umass_average": -1.4328882390292266,
            "umass_per_topic": {
                "topic_0": -1.4328882390292266,
                "topic_1": -1.1234567890123456,
                "topic_2": -0.9876543210987654
                }
        }
    },
  "topic_dist_img": "<matplotlib plot object>",
  "topic_document_counts": [____]
}
```

### Command Line Usage
```bash
# Turkish text analysis
manta-topic-modelling analyze data.csv --column text --language TR --topics 5

# English text analysis with lemmatization and visualizations
manta-topic-modelling analyze data.csv --column content --language EN --topics 10 --lemmatize --wordclouds --excel

# Custom tokenizer for Turkish text
manta-topic-modelling analyze reviews.csv --column review_text --language TR --topics 8 --tokenizer bpe --wordclouds

# NMTF analysis for topic relationship discovery
manta-topic-modelling analyze data.csv --column text --language TR --topics 5 --nmf-method nmtf

# Filter by app name and country
manta-topic-modelling analyze reviews.csv --column REVIEW --language TR --topics 5 --filter-app MyApp --filter-country TR

# Custom filtering columns
manta-topic-modelling analyze data.csv --column text --language TR --topics 5 --filter-app-column APP_ID --filter-country-column REGION

# Disable emoji processing for faster processing
manta-topic-modelling analyze data.csv --column text --language EN --topics 5 --emoji-map False

# Generate t-SNE visualization for document-topic relationships
manta-topic-modelling analyze data.csv --column text --language EN --topics 10 --tsne-plot

# Generate interactive LDAvis-style visualization
manta-topic-modelling analyze data.csv --column text --language EN --topics 10 --ldavis-plot

# Time-series t-SNE visualization
manta-topic-modelling analyze reviews.csv --column REVIEW --language TR --topics 5 \
    --tsne-plot --tsne-time-column year --tsne-time-ranges "2020,2021,2022,2023" --tsne-cumulative

# N-gram discovery for better phrase detection
manta-topic-modelling analyze papers.csv --column abstract --language EN --topics 10 --n-grams-to-discover 200

# Auto-calculate n-gram count based on vocabulary size
manta-topic-modelling analyze papers.csv --column abstract --language EN --topics 10 --n-grams-auto --n-grams-auto-k 0.5

# Keep numbers for BPE merging (e.g., "covid19", "120mg")
manta-topic-modelling analyze medical.csv --column text --language EN --topics 8 --keep-numbers

# PageRank-weighted TF-IDF for citation-aware analysis
manta-topic-modelling analyze papers.csv --column abstract --language EN --topics 10 --pagerank-column pagerank_score
```

## Package Structure

```
manta/
├── _functions/
│   ├── common_language/          # Shared functionality across languages
│   │   ├── emoji_processor.py    # Emoji handling utilities
│   │   └── topic_extractor.py    # Cross-language topic analysis and extraction
│   ├── english/                  # English text processing modules
│   │   ├── english_entry.py             # English text processing entry point
│   │   ├── english_preprocessor.py      # Text cleaning and preprocessing
│   │   ├── english_vocabulary.py        # Vocabulary creation
│   │   ├── english_text_encoder.py      # Text-to-numerical conversion
│   │   ├── english_topic_analyzer.py    # Topic extraction utilities
│   │   ├── english_topic_output.py      # Topic visualization and output
│   │   └── english_nmf_core.py          # NMF implementation for English
│   ├── nmf/                      # NMF algorithm implementations
│   │   ├── nmf_orchestrator.py          # Main NMF interface
│   │   ├── nmf_initialization.py        # Matrix initialization strategies
│   │   ├── nmf_basic.py                 # Standard NMF algorithm
│   │   ├── nmf_projective_basic.py      # Basic projective NMF
│   │   ├── nmf_projective_enhanced.py   # Enhanced projective NMF
│   │   └── nmtf/                        # Non-negative Matrix Tri-Factorization
│   │       ├── nmtf.py                  # NMTF implementation with topic relationships
│   │       ├── nmtf_init.py             # NMTF initialization utilities
│   │       ├── nmtf_util.py             # NMTF helper functions
│   │       ├── extract_nmtf_topics.py   # Topic extraction for NMTF results
│   │       └── example_usage.py         # NMTF usage examples
│   ├── tfidf/                    # TF-IDF calculation modules
│   │   ├── tfidf_english_calculator.py  # English TF-IDF implementation
│   │   ├── tfidf_turkish_calculator.py  # Turkish TF-IDF implementation
│   │   ├── tfidf_tf_functions.py        # Term frequency functions
│   │   ├── tfidf_idf_functions.py       # Inverse document frequency functions
│   │   └── tfidf_bm25_turkish.py        # BM25 implementation for Turkish
│   └── turkish/                  # Turkish text processing modules
│       ├── turkish_entry.py             # Turkish text processing entry point
│       ├── turkish_preprocessor.py      # Turkish text cleaning
│       ├── turkish_tokenizer_factory.py # Tokenizer creation and training
│       ├── turkish_text_encoder.py      # Text-to-numerical conversion
│       └── turkish_tfidf_generator.py   # TF-IDF matrix generation
├── utils/                        # Helper utilities (organized into sub-modules)
│   ├── analysis/                       # Analysis utilities
│   │   ├── coherence_score.py              # Topic coherence evaluation
│   │   ├── distance_two_words.py           # Word distance calculation
│   │   ├── umass_test.py                   # UMass coherence testing
│   │   ├── word_cooccurrence.py            # Word co-occurrence analysis
│   │   └── word_cooccurrence_analyzer.py   # Advanced word co-occurrence analysis
│   ├── console/                        # Console management
│   │   └── console_manager.py              # Console and logging management utilities
│   ├── database/                       # Database utilities
│   │   ├── database_manager.py             # Database connection and management utilities
│   │   └── save_topics_db.py               # Topic database saving utilities
│   ├── export/                         # Export functionality
│   │   ├── export_excel.py                 # Excel export functionality
│   │   ├── json_to_excel.py                # JSON to Excel conversion utilities
│   │   ├── save_doc_score_pair.py          # Document-score pair saving utilities
│   │   └── save_word_score_pair.py         # Word-score pair saving utilities
│   ├── preprocess/                     # Preprocessing utilities
│   │   └── combine_number_suffix.py         # Number and suffix combination utilities
│   ├── visualization/                  # Visualization utilities
│   │   ├── gen_cloud.py                    # Word cloud generation
│   │   ├── image_to_base.py                # Image to base64 conversion
│   │   ├── topic_dist.py                   # Topic distribution plotting
│   │   └── visualizer.py                   # General visualization utilities
│   └── agent/                          # AI assistant utilities
│       ├── claude_prompt_generator.py       # Claude AI prompt generation utilities
│       └── claude_prompt_generator.html     # HTML interface for prompt generation
├── cli.py                        # Command-line interface
├── standalone_nmf.py             # Core NMF implementation
└── __init__.py                   # Package initialization and public API
```

## Installation

### From PyPI (Recommended)
```bash
pip install manta-topic-modelling
```

### From Source (Development)
1. Clone the repository:
```bash
git clone https://github.com/emirkyz/manta.git
cd manta
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The package provides the `manta-topic-modelling` command with an `analyze` subcommand:

```bash
# Basic usage
manta-topic-modelling analyze data.csv --column text --language TR --topics 5

# Advanced usage with all options
manta-topic-modelling analyze reviews.csv \
  --column review_text \
  --language EN \
  --topics 10 \
  --words-per-topic 20 \
  --nmf-method pnmf \
  --lemmatize \
  --wordclouds \
  --excel \
  --topic-distribution \
  --output-name my_analysis
```

#### Command Line Options

**Required Arguments:**
- `filepath`: Path to input CSV or Excel file
- `--column, -c`: Name of column containing text data
- `--language, -l`: Language ("TR" for Turkish, "EN" for English)

**Optional Arguments:**
- `--topics, -t`: Number of topics to extract (default: 5)
- `--output-name, -o`: Custom name for output files (default: auto-generated)
- `--output-dir`: Directory to save output files (default: current directory)
- `--tokenizer`: Tokenizer type for Turkish ("bpe" or "wordpiece", default: "bpe")
- `--nmf-method`: Factorization algorithm ("nmf", "pnmf", or "nmtf", default: "nmf")
- `--words-per-topic`: Number of top words per topic (default: 15)
- `--lemmatize`: Apply lemmatization for English text
- `--emoji-map`: Enable emoji processing and mapping (default: True). Use --emoji-map False to disable
- `--wordclouds`: Generate word cloud visualizations
- `--excel`: Export results to Excel format
- `--topic-distribution`: Generate topic distribution plots
- `--separator`: CSV separator character (default: ",")
- `--filter-app`: Filter data by specific app name
- `--filter-app-column`: Column name for app filtering (default: "PACKAGE_NAME")
- `--filter-country`: Filter data by country code (e.g., TR, US, GB)
- `--filter-country-column`: Column name for country filtering (default: "COUNTRY")

**Visualization Options:**
- `--tsne-plot`: Generate t-SNE 2D visualization of document-topic relationships
- `--tsne-time-column`: Column name for time-series t-SNE visualization
- `--tsne-time-ranges`: Comma-separated time ranges (e.g., "2020,2021,2022,2023")
- `--tsne-cumulative`: Use cumulative time periods (show data "up to year X")
- `--ldavis-plot`: Generate interactive LDAvis-style topic exploration

**N-gram Discovery Options:**
- `--n-grams-to-discover`: Number of n-grams to discover via BPE (English only)
- `--n-grams-auto`: Auto-calculate n-gram count based on vocabulary size
- `--n-grams-auto-k`: Scaling constant k for auto n-gram formula (default: 0.5)
- `--keep-numbers`: Preserve numbers during preprocessing for BPE merging
- `--no-pmi`: Disable PMI scoring for BPE when --keep-numbers is used

**Advanced Options:**
- `--pagerank-column`: Column with PageRank scores for TF-IDF weighting (boosts high-PageRank docs 1-2x)
- `--word-pairs`: Generate word co-occurrence analysis and heatmap
- `--save-to-db`: Save data to database for persistence

### Python API

```python
from manta import run_topic_analysis

# Basic English text analysis
results = run_topic_analysis(
    filepath="data.csv",
    column="review_text",
    language="EN",
    topic_count=5,
    lemmatize=True,
    generate_wordclouds=True,
    export_excel=True
)

# Advanced Turkish text analysis with filtering
results = run_topic_analysis(
    filepath="turkish_reviews.csv",
    column="yorum_metni",
    language="TR",
    topic_count=10,
    words_per_topic=15,
    tokenizer_type="bpe",
    nmf_method="nmf",
    generate_wordclouds=True,
    export_excel=True,
    topic_distribution=True,
    filter_app=True,
    data_filter_options={
        "filter_app_name": "MyApp",
        "filter_app_column": "APP_NAME", 
        "filter_app_country": "TR",
        "filter_app_country_column": "COUNTRY_CODE"
    }
)
```

#### API Parameters

**Required (one of):**
- `filepath` (str): Path to input CSV or Excel file
- `dataframe` (DataFrame): Pandas DataFrame containing text data (alternative to filepath)
- `column` (str): Name of column containing text data

**Optional:**
- `separator` (str): CSV separator character (default: ",")
- `language` (str): "TR" for Turkish, "EN" for English (default: "EN")
- `topic_count` (int): Number of topics to extract (default: 5). Set to -1 for auto-selection.
- `nmf_method` (str): "nmf", "pnmf", or "nmtf" algorithm variant (default: "nmf")
- `lemmatize` (bool): Apply lemmatization for English (default: False)
- `tokenizer_type` (str): "bpe" or "wordpiece" for Turkish (default: "bpe")
- `words_per_topic` (int): Top words to show per topic (default: 15)
- `word_pairs_out` (bool): Create word pairs output (default: True)
- `generate_wordclouds` (bool): Create word cloud visualizations (default: True)
- `export_excel` (bool): Export results to Excel (default: True)
- `topic_distribution` (bool): Generate distribution plots (default: True)
- `filter_app` (bool): Enable app filtering (default: False)
- `data_filter_options` (dict): Advanced filtering options with keys (all default to empty string):
  - `filter_app_name` (str): App name for filtering
  - `filter_app_column` (str): Column name for app filtering (default: "PACKAGE_NAME")
  - `filter_app_country` (str): Country code for filtering (case-insensitive)
  - `filter_app_country_column` (str): Column name for country filtering (default: "COUNTRY")
- `emoji_map` (bool): Enable emoji processing and mapping (default: False)
- `output_name` (str): Custom output directory name (default: auto-generated)
- `save_to_db` (bool): Whether to persist data to database (default: False)
- `output_dir` (str): Base directory for outputs (default: current working directory)
- `pagerank_column` (str): Column with PageRank scores for TF-IDF weighting (boosts high-PageRank docs 1-2x)

**N-gram Discovery (via kwargs):**
- `n_grams_to_discover` (int/str): Number of n-grams to discover via BPE, or "auto" for automatic (default: None)
- `ngram_auto_k` (float): Scaling constant for auto n-gram formula: sqrt(vocab_size) * k (default: 0.5)

**Visualization Options (via kwargs):**
- `gen_tsne` (bool): Generate t-SNE 2D visualization of document-topic relationships (default: False)
- `gen_ldavis_plot` (bool): Generate interactive LDAvis-style topic exploration (default: False)
- `tsne_time_column` (str): Column for time-series t-SNE visualization (default: None)
- `tsne_time_ranges` (list): Time ranges for time-series visualization (default: None)
- `tsne_cumulative` (bool): Use cumulative time periods in t-SNE (default: False)

**Co-occurrence Analysis (via kwargs):**
- `cooccurrence_window_size` (int): Window size for word co-occurrence (default: 5)
- `cooccurrence_min_count` (int): Minimum co-occurrence count (default: 2)
- `cooccurrence_top_n` (int): Top n word pairs to display (default: 100)

**Cache/Processing Options (via kwargs):**
- `use_cache` (bool): Check for cached preprocessed data (default: True)
- `force_reprocess` (bool): Force reprocessing, ignore cache (default: False)
- `nmf_variants` (List[str]): List of NMF variants to run (default: None)

## Outputs

The analysis generates several outputs in an `Output/` directory (created at runtime), organized in a subdirectory named after your analysis:

- **Topic-Word Excel File**: `.xlsx` file containing top words for each topic and their scores
- **Word Clouds**: PNG images of word clouds for each topic (if `generate_wordclouds=True`)
- **Topic Distribution Plot**: Plot showing distribution of documents across topics (if `topic_distribution=True`)
- **Coherence Scores**: JSON file with coherence scores for the topics
- **Top Documents**: JSON file listing most representative documents for each topic

## Features

- **Multi-language Support**: Optimized processing for both Turkish and English texts
- **Advanced Tokenization**: BPE and WordPiece tokenizers for Turkish, traditional tokenization for English
- **Multiple Factorization Algorithms**: Standard NMF, Orthogonal Projective NMF (PNMF), and Non-negative Matrix Tri-Factorization (NMTF)
- **Advanced NMF Variants**: Hierarchical NMF, Online NMF, and Symmetric NMF implementations
- **N-gram Discovery**: Automatic discovery of meaningful word combinations using BPE for English text
- **Rich Visualizations**: Word clouds and topic distribution plots
- **Flexible Export**: Excel and JSON export formats with organized export utilities
- **Coherence Evaluation**: Built-in topic coherence scoring and advanced analysis tools
- **Database Management**: Comprehensive SQLite database integration with dedicated management utilities
- **Modular Architecture**: Organized utility modules for analysis, visualization, export, and preprocessing
- **Text Preprocessing**: Language-specific text cleaning and preprocessing

### N-gram Discovery

MANTA supports automatic n-gram discovery using BPE (Byte Pair Encoding) for English text. This feature identifies frequently occurring word combinations and adds them to the vocabulary as new tokens.

To enable n-gram discovery, use the `n_grams_to_discover` parameter:

```python
results = run_topic_analysis(
    filepath="data.csv",
    column="text",
    language="EN",
    n_grams_to_discover=200  # Discover 200 word combinations
)
```

This can improve topic quality by capturing meaningful phrases like "machine_learning" or "climate_change" as single tokens.

### Advanced Visualization Options

MANTA provides multiple visualization options for exploring topics and document relationships.

#### t-SNE Visualization

Generate a 2D t-SNE visualization showing document-topic relationships:

```python
results = run_topic_analysis(
    filepath="data.csv",
    column="text",
    language="EN",
    topic_count=10,
    gen_tsne=True
)
```

For time-series analysis, use t-SNE with time ranges:

```python
results = run_topic_analysis(
    filepath="reviews.csv",
    column="review_text",
    language="EN",
    topic_count=8,
    gen_tsne=True,
    tsne_time_column="year",
    tsne_time_ranges=["2020", "2021", "2022", "2023"],
    tsne_cumulative=True  # Show cumulative data up to each year
)
```

#### LDAvis Interactive Visualization

Generate an interactive LDAvis-style HTML visualization for topic exploration:

```python
results = run_topic_analysis(
    filepath="documents.csv",
    column="content",
    language="EN",
    topic_count=12,
    gen_ldavis_plot=True
)
```

This creates an interactive HTML file where you can:
- Explore topic relationships in 2D space
- See top words for each topic with relevance adjustment
- Understand term-topic distributions

## Requirements

- Python 3.9+
- Dependencies are automatically installed with the package

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the [GitHub repository](https://github.com/emirkyz/manta/issues?q=is%3Aissue)
