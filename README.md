# NMF Standalone

[![PyPI version](https://badge.fury.io/py/nmf-standalone.svg)](https://badge.fury.io/py/nmf-standalone)
[![PyPI version](https://img.shields.io/pypi/v/nmf-standalone)](https://badge.fury.io/py/nmf-standalone)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive topic modeling system using Non-negative Matrix Factorization (NMF) that supports both English and Turkish text processing. Features advanced tokenization techniques, multiple NMF algorithms, and rich visualization capabilities.

## Quick Start
### Local Dev
To build and run the app locally for development:
```bash
uv pip install -e .
pip install -e .
```
After that you can import and use the app.

### Installation from PyPI
```bash
pip install nmf-standalone
```

### Command Line Usage
```bash
# Turkish text analysis
nmf-standalone analyze data.csv --column text --language TR --topics 5

# English text analysis with lemmatization and visualizations
nmf-standalone analyze data.csv --column content --language EN --topics 10 --lemmatize --wordclouds --excel

# Custom tokenizer for Turkish text
nmf-standalone analyze reviews.csv --column review_text --language TR --topics 8 --tokenizer bpe --wordclouds
```

### Python API Usage
```python
from nmf_standalone import run_topic_analysis

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

## Package Structure

```
nmf_standalone/
├── functions/
│   ├── common_language/          # Shared functionality across languages
│   │   ├── emoji_processor.py    # Emoji handling utilities
│   │   └── topic_analyzer.py     # Cross-language topic analysis
│   ├── english/                  # English text processing modules
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
│       ├── turkish_preprocessor.py      # Turkish text cleaning
│       ├── turkish_tokenizer_factory.py # Tokenizer creation and training
│       ├── turkish_text_encoder.py      # Text-to-numerical conversion
│       └── turkish_tfidf_generator.py   # TF-IDF matrix generation
├── utils/                        # Helper utilities
│   ├── coherence_score.py              # Topic coherence evaluation
│   ├── gen_cloud.py                    # Word cloud generation
│   ├── export_excel.py                 # Excel export functionality
│   ├── topic_dist.py                   # Topic distribution plotting
│   └── other/                           # Additional utility functions
├── cli.py                        # Command-line interface
├── standalone_nmf.py             # Core NMF implementation
└── __init__.py                   # Package initialization and public API
```

## Installation

### From PyPI (Recommended)
```bash
pip install nmf-standalone
```

### From Source (Development)
1. Clone the repository:
```bash
git clone https://github.com/emirkyz/nmf-standalone.git
cd nmf-standalone
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

The package provides the `nmf-standalone` command with an `analyze` subcommand:

```bash
# Basic usage
nmf-standalone analyze data.csv --column text --language TR --topics 5

# Advanced usage with all options
nmf-standalone analyze reviews.csv \
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
- `--wordclouds`: Generate word cloud visualizations
- `--excel`: Export results to Excel format
- `--topic-distribution`: Generate topic distribution plots
- `--separator`: CSV separator character (default: "|")
- `--filter-app`: Filter data by specific app name

### Python API

```python
from nmf_standalone import run_topic_analysis

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

For issues and questions, please open an issue on the [GitHub repository](https://github.com/emirkyz/nmf-standalone/issues?q=is%3Aissue)