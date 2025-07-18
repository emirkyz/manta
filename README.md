# MANTA (Multi-lingual Advanced NMF-based Topic Analysis)

[![PyPI version](https://badge.fury.io/py/manta-topic-modelling.svg)](https://badge.fury.io/py/manta-topic-modelling)
[![PyPI version](https://img.shields.io/pypi/v/manta-topic-modelling)](https://badge.fury.io/py/manta-topic-modelling)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive topic modeling system using Non-negative Matrix Factorization (NMF) that supports both English and Turkish text processing. Features advanced tokenization techniques, multiple NMF algorithms, and rich visualization capabilities.

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

### Command Line Usage
```bash
# Turkish text analysis
manta-topic-modelling analyze data.csv --column text --language TR --topics 5

# English text analysis with lemmatization and visualizations
manta-topic-modelling analyze data.csv --column content --language EN --topics 10 --lemmatize --wordclouds --excel

# Custom tokenizer for Turkish text
manta-topic-modelling analyze reviews.csv --column review_text --language TR --topics 8 --tokenizer bpe --wordclouds

# Disable emoji processing for faster processing
manta-topic-modelling analyze data.csv --column text --language EN --topics 5 --emoji-map False
```

### Python API Usage
```python
from manta import run_topic_analysis

# Simple topic modeling
results = run_topic_analysis(
    filepath="data.csv",
    column="review_text",
    language="EN",
    topics=5,
    lemmatize=True
)

# Turkish text analysis
results = run_topic_analysis(
    filepath="turkish_reviews.csv", 
    column="yorum_metni",
    language="TR",
    topics=8,
    tokenizer_type="bpe",
    generate_wordclouds=True
)
```

## Result Structre
```
{
"state": State of the analysis, either "success" or "error",
"message": Message about the result of the analysis,
"data_name": Name of the input data file,
"topic_word_scores": JSON object containing topics and their top words with scores,
"topic_doc_scores": JSON object containing topics and their top documents with scores,
"coherence_scores": JSON object containing coherence scores for each topic,
"topic_dist_img": Matplplotlib plt object of topic distribution plot if `gen_topic_distribution` is True,
"topic_document_counts": Count of documents per topic,
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

## Package Structure

```
manta/
├── _functions/
│   ├── common_language/          # Shared functionality across languages
│   │   ├── emoji_processor.py    # Emoji handling utilities
│   │   └── topic_analyzer.py     # Cross-language topic analysis
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
│   │   └── nmf_projective_enhanced.py   # Enhanced projective NMF
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
├── utils/                        # Helper utilities
│   ├── coherence_score.py              # Topic coherence evaluation
│   ├── combine_number_suffix.py         # Number and suffix combination utilities
│   ├── distance_two_words.py           # Word distance calculation
│   ├── export_excel.py                 # Excel export functionality
│   ├── gen_cloud.py                    # Word cloud generation
│   ├── hierarchy_nmf.py                # Hierarchical NMF utilities
│   ├── image_to_base.py                # Image to base64 conversion
│   ├── save_doc_score_pair.py          # Document-score pair saving utilities
│   ├── save_topics_db.py               # Topic database saving
│   ├── save_word_score_pair.py         # Word-score pair saving utilities
│   ├── topic_dist.py                   # Topic distribution plotting
│   ├── umass_test.py                   # UMass coherence testing
│   ├── visualizer.py                   # General visualization utilities
│   ├── word_cooccurrence.py            # Word co-occurrence analysis
│   └── other/                           # Additional utility functions
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
  --nmf-method opnmf \
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
- `--tokenizer`: Tokenizer type for Turkish ("bpe" or "wordpiece", default: "bpe")
- `--nmf-method`: NMF algorithm ("nmf" or "opnmf", default: "nmf")
- `--words-per-topic`: Number of top words per topic (default: 15)
- `--lemmatize`: Apply lemmatization for English text
- `--emoji-map`: Enable emoji processing and mapping (default: True). Use --emoji-map False to disable
- `--wordclouds`: Generate word cloud visualizations
- `--excel`: Export results to Excel format
- `--topic-distribution`: Generate topic distribution plots
- `--separator`: CSV separator character (default: "|")
- `--filter-app`: Filter data by specific app name

### Python API

```python
from manta import run_topic_analysis

# Basic English text analysis
results = run_topic_analysis(
    filepath="data.csv",
    column="review_text",
    language="EN",
    topics=5,
    lemmatize=True,
    generate_wordclouds=True,
    export_excel=True
)

# Advanced Turkish text analysis
results = run_topic_analysis(
    filepath="turkish_reviews.csv",
    column="yorum_metni",
    language="TR",
    topics=10,
    words_per_topic=15,
    tokenizer_type="bpe",
    nmf_method="nmf",
    generate_wordclouds=True,
    export_excel=True,
    topic_distribution=True
)
```

#### API Parameters

**Required:**
- `filepath` (str): Path to input CSV or Excel file
- `column` (str): Name of column containing text data

**Optional:**
- `language` (str): "TR" for Turkish, "EN" for English (default: "EN")
- `topics` (int): Number of topics to extract (default: 5)
- `words_per_topic` (int): Top words to show per topic (default: 15)
- `nmf_method` (str): "nmf" or "opnmf" algorithm variant (default: "nmf")
- `tokenizer_type` (str): "bpe" or "wordpiece" for Turkish (default: "bpe")
- `lemmatize` (bool): Apply lemmatization for English (default: True)
- `generate_wordclouds` (bool): Create word cloud visualizations (default: True)
- `export_excel` (bool): Export results to Excel (default: True)
- `topic_distribution` (bool): Generate distribution plots (default: True)
- `emoji_map` (bool): Enable emoji processing and mapping (default: True)
- `output_name` (str): Custom output directory name (default: auto-generated)
- `separator` (str): CSV separator character (default: ",")
- `filter_app` (bool): Enable app filtering (default: False)
- `filter_app_name` (str): App name for filtering (default: "")

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
- **Multiple NMF Algorithms**: Standard NMF and Orthogonal Projective NMF (OPNMF)
- **Rich Visualizations**: Word clouds and topic distribution plots
- **Flexible Export**: Excel and JSON export formats
- **Coherence Evaluation**: Built-in topic coherence scoring
- **Text Preprocessing**: Language-specific text cleaning and preprocessing

## Requirements

- Python 3.9+
- Dependencies are automatically installed with the package

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the [GitHub repository](https://github.com/emirkyz/manta/issues?q=is%3Aissue)