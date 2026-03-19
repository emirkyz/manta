# MANTA — Multi-lingual Advanced NMF-based Topic Analysis

[![PyPI version](https://badge.fury.io/py/manta-topic-modelling.svg)](https://badge.fury.io/py/manta-topic-modelling)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MANTA is an open-source Python library for topic modeling using Non-negative Matrix Factorization (NMF), Projective Non-Negative Matrix Factorization and Non-negative Matrix Tri-Factorization (NMTF). It primarily supports **English** text processing, with **Turkish** support actively being improve. It offers advanced tokenization, multiple term-weighting schemes, and rich visualization capabilities — all through a simple one-function interface.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Python API](#python-api)
  - [Parameters](#api-parameters)
  - [Result Structure](#result-structure)
- [Command Line Interface](#command-line-interface)
  - [CLI Options](#cli-options)
- [Key Features](#key-features)
  - [N-gram Discovery](#n-gram-discovery)
  - [Visualization](#visualization)
  - [PageRank-weighted TF-IDF](#pagerank-weighted-tf-idf)
- [Outputs](#outputs)
- [Package Structure](#package-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Installation

**From PyPI (recommended):**

```bash
pip install manta-topic-modelling
```
or using uv:
```bash
uv add manta-topic-modelling
```

**From source (development):**

```bash
git clone https://github.com/emirkyz/manta.git
cd manta
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

---

## Quick Start

```python
from manta import run_topic_analysis

results = run_topic_analysis(
    filepath="data.csv",
    column="review_text",
    language="EN",
    topic_count=5,
    lemmatize=True
)
```

That's it. MANTA handles preprocessing, tokenization, matrix factorization, and output generation automatically.

---

## Python API

### Basic Examples

```python
from manta import run_topic_analysis

# English analysis with visualizations
results = run_topic_analysis(
    filepath="data.csv",
    column="review_text",
    language="EN",
    topic_count=5,
    lemmatize=True,
    generate_wordclouds=True,
    export_excel=True
)

# Turkish analysis with BPE tokenizer
results = run_topic_analysis(
    filepath="turkish_reviews.csv",
    column="yorum_metni",
    language="TR",
    topic_count=8,
    tokenizer_type="bpe",
    generate_wordclouds=True
)

# Pass a DataFrame directly instead of a file path
import pandas as pd
df = pd.read_csv("reviews.csv")
results = run_topic_analysis(
    dataframe=df,
    column="review_text",
    language="EN",
    topic_count=5
)
```
### Advanced Examples
 
```python
# NMTF for topic relationship discovery
results = run_topic_analysis(
    filepath="data.csv",
    column="text_content",
    language="TR",
    topic_count=6,
    nmf_method="nmtf",
    generate_wordclouds=True
)
 
# t-SNE + LDAvis visualizations
results = run_topic_analysis(
    filepath="research_papers.csv",
    column="abstract",
    language="EN",
    topic_count=10,
    gen_tsne=True,
    gen_ldavis_plot=True,
    generate_wordclouds=True
)
 
# N-gram discovery for better phrase detection
results = run_topic_analysis(
    filepath="papers.csv",
    column="abstract",
    language="EN",
    topic_count=10,
    n_grams_to_discover=200  # Captures phrases like "machine_learning"
)
 
# PageRank-weighted TF-IDF for citation-aware modeling
results = run_topic_analysis(
    filepath="papers_with_pagerank.csv",
    column="abstract",
    language="EN",
    topic_count=8,
    pagerank_column="pagerank_score"
)
 
# Turkish analysis with app/country filtering
results = run_topic_analysis(
    filepath="turkish_reviews.csv",
    column="yorum_metni",
    language="TR",
    topic_count=10,
    words_per_topic=15,
    tokenizer_type="bpe",
    filter_app=True,
    data_filter_options={
        "filter_app_name": "MyApp",
        "filter_app_column": "APP_NAME",
        "filter_app_country": "TR",
        "filter_app_country_column": "COUNTRY_CODE"
    }
)
```

### API Parameters

**Data Input (one required):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | str | Path to input CSV or Excel file |
| `dataframe` | DataFrame | Pandas DataFrame (alternative to filepath) |
| `column` | str | Column containing text data |

**Core Settings:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | str | `"EN"` | `"TR"` for Turkish, `"EN"` for English |
| `topic_count` | int | `5` | Number of topics (`-1` for auto-selection) |
| `nmf_method` | str | `"nmf"` | Algorithm: `"nmf"`, `"pnmf"`, or `"nmtf"` |
| `lemmatize` | bool | `False` | Apply lemmatization (English only) |
| `tokenizer_type` | str | `"bpe"` | Tokenizer for Turkish: `"bpe"` or `"wordpiece"` |
| `words_per_topic` | int | `15` | Top words shown per topic |
| `separator` | str | `","` | CSV separator character |

**Output & Export:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generate_wordclouds` | bool | `True` | Create word cloud images |
| `export_excel` | bool | `True` | Export results to `.xlsx` |
| `topic_distribution` | bool | `True` | Generate distribution plots |
| `output_name` | str | auto | Custom output directory name |
| `output_dir` | str | cwd | Base directory for outputs |
| `save_to_db` | bool | `False` | Persist data to SQLite database |

**Visualization (via kwargs):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gen_tsne` | bool | `False` | Generate t-SNE 2D visualization |
| `gen_ldavis_plot` | bool | `False` | Generate interactive LDAvis HTML |


**N-gram Discovery (via kwargs):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_grams_to_discover` | int/str | `None` | N-grams to discover via BPE, or `"auto"` |
| `ngram_auto_k` | float | `0.5` | Scaling constant for auto formula: `sqrt(vocab_size) * k` |

**Filtering:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_app` | bool | `False` | Enable app filtering |
| `data_filter_options` | dict | `{}` | Keys: `filter_app_name`, `filter_app_column`, `filter_app_country`, `filter_app_country_column` |
| `emoji_map` | bool | `False` | Enable emoji processing and mapping |
| `pagerank_column` | str | `None` | Column with PageRank scores (boosts high-PageRank docs 1–2×) |

**Advanced (via kwargs):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cooccurrence_window_size` | int | `5` | Window size for co-occurrence |
| `cooccurrence_min_count` | int | `2` | Minimum co-occurrence count |
| `cooccurrence_top_n` | int | `100` | Top word pairs to display |
| `use_cache` | bool | `True` | Check for cached preprocessed data |
| `force_reprocess` | bool | `False` | Force reprocessing, ignore cache |
| `nmf_variants` | list | `None` | List of NMF variants to run |

### Result Structure

`run_topic_analysis` returns a dictionary:

```python
{
    "state": "success",                  # "success" or "error"
    "message": "Analysis completed successfully",
    "data_name": "reviews.csv",
    "topic_word_scores": {               # Top words per topic with scores
        "topic_0": {"word1": 0.15, "word2": 0.12, ...}
    },
    "topic_doc_scores": {                # Top documents per topic with scores
        "topic_0": [{"document": "Sample text...", "score": 0.78}]
    },
    "coherence_scores": {                # Coherence evaluation
        "gensim": {
            "umass_average": -1.43,
            "umass_per_topic": {"topic_0": -1.43, ...}
        }
    },
    "topic_dist_img": "<matplotlib object>",  # If gen_topic_distribution=True
    "topic_document_counts": [...],
    "topic_relationships": ...           # Topic-to-topic matrix (NMTF only)
}
```

---

## Command Line Interface

MANTA provides the `manta-topic-modelling` command with an `analyze` subcommand.

### Basic Usage

```bash
# English analysis
manta-topic-modelling analyze data.csv --column content --language EN --topics 10 --lemmatize --wordclouds --excel

# Turkish analysis
manta-topic-modelling analyze data.csv --column text --language TR --topics 5
```

### More Examples

```bash
# NMTF for topic relationships
manta-topic-modelling analyze data.csv --column text --language TR --topics 5 --nmf-method nmtf

# t-SNE + LDAvis visualizations
manta-topic-modelling analyze data.csv --column text --language EN --topics 10 --tsne-plot --ldavis-plot

# N-gram discovery
manta-topic-modelling analyze papers.csv --column abstract --language EN --topics 10 --n-grams-to-discover 200

# Auto n-grams based on vocabulary size
manta-topic-modelling analyze papers.csv --column abstract --language EN --topics 10 --n-grams-auto --n-grams-auto-k 0.5

# Filter by app and country
manta-topic-modelling analyze reviews.csv --column REVIEW --language TR --topics 5 --filter-app MyApp --filter-country TR

# PageRank-weighted TF-IDF
manta-topic-modelling analyze papers.csv --column abstract --language EN --topics 10 --pagerank-column pagerank_score
```

### CLI Options

**Required:**

| Argument | Description |
|----------|-------------|
| `filepath` | Path to input CSV or Excel file |
| `--column, -c` | Column containing text data |
| `--language, -l` | `"TR"` or `"EN"` |

**Analysis:**

| Option | Default | Description |
|--------|---------|-------------|
| `--topics, -t` | 5 | Number of topics |
| `--nmf-method` | nmf | `"nmf"`, `"pnmf"`, or `"nmtf"` |
| `--tokenizer` | bpe | Turkish tokenizer: `"bpe"` or `"wordpiece"` |
| `--words-per-topic` | 15 | Top words per topic |
| `--lemmatize` | off | Apply English lemmatization |
| `--emoji-map` | True | Emoji processing (use `--emoji-map False` to disable) |
| `--keep-numbers` | off | Preserve numbers for BPE merging |
| `--no-pmi` | off | Disable PMI scoring for BPE |

**Output:**

| Option | Description |
|--------|-------------|
| `--output-name, -o` | Custom output name (default: auto) |
| `--output-dir` | Output directory (default: cwd) |
| `--wordclouds` | Generate word clouds |
| `--excel` | Export to Excel |
| `--topic-distribution` | Generate distribution plots |

**Visualization:**

| Option | Description |
|--------|-------------|
| `--tsne-plot` | Generate t-SNE 2D visualization |
| `--ldavis-plot` | Interactive LDAvis HTML |

**N-gram Discovery:**

| Option | Description |
|--------|-------------|
| `--n-grams-to-discover` | Number of n-grams via BPE |
| `--n-grams-auto` | Auto-calculate from vocab size |
| `--n-grams-auto-k` | Scaling constant (default: 0.5) |

**Filtering:**

| Option | Description |
|--------|-------------|
| `--filter-app` | Filter by app name |
| `--filter-app-column` | App column name (default: `PACKAGE_NAME`) |
| `--filter-country` | Filter by country code |
| `--filter-country-column` | Country column name (default: `COUNTRY`) |
| `--separator` | CSV separator (default: `,`) |

**Advanced:**

| Option | Description |
|--------|-------------|
| `--pagerank-column` | PageRank scores column for TF-IDF weighting |
| `--word-pairs` | Generate co-occurrence heatmap |
| `--save-to-db` | Persist data to database |

---

## Key Features

### Multi-language Support

MANTA is optimized for both **English** and **Turkish** text. Turkish support includes corpus-specific BPE and WordPiece tokenizers that handle the language's rich morphology, while English processing supports lemmatization and traditional tokenization.

### Multiple Factorization Algorithms

- **NMF** — Standard Non-negative Matrix Factorization
- **PNMF** — Orthogonal Projective NMF for more distinct topics
- **NMTF** — Non-negative Matrix Tri-Factorization, which also produces a topic-to-topic relationship matrix

### N-gram Discovery

MANTA can automatically discover meaningful multi-word phrases using BPE (Byte Pair Encoding) inspired method for English text. This captures phrases like "machine_learning" or "climate_change" as single tokens, improving topic quality. Additionally, MANTA doesn't limit the length of the n-gram word, meaning as long as the n_gram_to_discover parameter is high enough, MANTA can extract up to 7 grams and more.

```python
results = run_topic_analysis(
    filepath="data.csv",
    column="text",
    language="EN",
    n_grams_to_discover=200
)
```

### Visualization

**t-SNE** — 2D visualization of document-topic relationships:

```python
results = run_topic_analysis(
    filepath="reviews.csv",
    column="review_text",
    language="EN",
    topic_count=8,
    gen_tsne=True,
    tsne_cumulative=True
)
```

**LDAvis** — Interactive HTML visualization for exploring topics, top words, and term-topic distributions:

```python
results = run_topic_analysis(
    filepath="documents.csv",
    column="content",
    language="EN",
    topic_count=12,
    gen_ldavis_plot=True
)
```

**Word Clouds** — Per-topic word cloud images generated automatically when `generate_wordclouds=True`.

### PageRank-weighted TF-IDF

For citation-aware topic modeling, provide a column of PageRank scores. Documents with higher scores are boosted 1–2× in the TF-IDF matrix, giving more influence to authoritative documents.

---

## Outputs

Analysis results are saved to an `Output/` directory, organized in a subdirectory named after your analysis:

- **Topic-Word Excel** — `.xlsx` with top words and scores per topic
- **Word Clouds** — PNG images per topic
- **Topic Distribution Plot** — Document distribution across topics
- **Coherence Scores** — JSON with UMass coherence scores
- **Top Documents** — JSON with most representative documents per topic
- **LDAvis HTML** — Interactive visualization (if enabled)
- **t-SNE Plot** — 2D document-topic map (if enabled)

---

## Package Structure

```
manta/
├── __init__.py                   # Public API
├── cli.py                        # Command-line interface
├── standalone_nmf.py             # Core NMF implementation
├── _functions/
│   ├── common_language/          # Shared cross-language utilities
│   ├── english/                  # English processing pipeline
│   ├── turkish/                  # Turkish processing pipeline
│   ├── nmf/                      # NMF algorithms (NMF & PNMF)
│   │   └── nmtf/                 # Tri-Factorization
│   └── tfidf/                    # TF-IDF modules
│       ├── tfidf_english_calculator.py
│       ├── tfidf_turkish_calculator.py
│       ├── tfidf_tf_functions.py
│       ├── tfidf_idf_functions.py
│       └── tfidf_bm25_turkish.py
└── utils/
    ├── analysis/                 # Coherence, co-occurrence
    ├── console/                  # Logging
    ├── database/                 # SQLite persistence
    ├── export/                   # Excel, JSON export
    ├── preprocess/               # Number/suffix utilities
    ├── visualization/            # Word clouds, t-SNE, plots
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request on the [GitHub repository](https://github.com/emirkyz/manta).

For issues and questions, [open an issue](https://github.com/emirkyz/manta/issues).

---

## Citation

If you use MANTA in your research, please cite:

```bibtex
@article{KARAYAGIZ2025102386,
  title     = {Manta: Multi-lingual advanced NMF-based topic analysis},
  journal   = {SoftwareX},
  volume    = {32},
  pages     = {102386},
  year      = {2025},
  issn      = {2352-7110},
  doi       = {https://doi.org/10.1016/j.softx.2025.102386},
  url       = {https://www.sciencedirect.com/science/article/pii/S2352711025003528},
  author    = {Emir Karayağız and Tolga Berber},
  keywords  = {Topic modeling, Non-negative matrix factorization, Python, Natural language processing, Information retrieval},
  abstract  = {This paper presents MANTA (Multi-lingual Advanced NMF-based Topic Analysis), a novel open-source Python library that provides an integrated pipeline to address key limitations in existing topic modeling workflows. MANTA provides an integrated, easy-to-use pipeline for Non-negative Matrix Factorization (NMF) based topic analysis, uniquely combining corpus-specific subword tokenization (BPE/WordPiece) with advanced term weighting schemes (SMART, BM25) and flexible NMF solver options, including a high-performance Projective NMF method. It offers native support for both English and morphologically complex languages like Turkish. With a simple one-function interface and a command-line utility, MANTA lowers the technical barrier for sophisticated topic analysis, making it a powerful tool for researchers in computational social science and digital humanities.}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
