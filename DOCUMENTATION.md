# MANTA (Multi-lingual Advanced NMF-based Topic Analysis) - Comprehensive Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Main Functions Documentation](#main-functions-documentation)
4. [Workflow Process](#workflow-process)
5. [Module Structure](#module-structure)
6. [Configuration Options](#configuration-options)
7. [Output Files](#output-files)
8. [Usage Examples](#usage-examples)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

**MANTA (Multi-lingual Advanced NMF-based Topic Analysis)** is a comprehensive topic modeling tool that uses **Non-negative Matrix Factorization (NMF)** and **Non-negative Matrix Tri-Factorization (NMTF)** to extract meaningful topics from text documents. The system supports both **Turkish** and **English** languages and provides end-to-end processing from raw text data to visualized topic results.

### Key Features
- **Bilingual Support**: Handles both Turkish and English text processing
- **Multiple Tokenization Methods**: BPE (Byte-Pair Encoding) and WordPiece for Turkish
- **Multiple NMF Factorization Algorithms**: Standard NMF, Projective NMF (PNMF), and Non-negative Matrix Tri-Factorization (NMTF)
- **Advanced NMF Variants**: Hierarchical NMF, Online NMF, and Symmetric NMF implementations
- **Rich Output Generation**: Word clouds, topic distributions, Excel reports, coherence scores, and topic relationship analysis (NMTF)
- **Database Management**: Comprehensive SQLite database integration with dedicated management utilities
- **Modular Architecture**: Organized utility modules for analysis, visualization, export, and preprocessing
- **Comprehensive Preprocessing**: Text cleaning, tokenization, and TF-IDF vectorization

### Use Cases
- **Academic Research**: Analyzing research papers, dissertations, and academic texts
- **App Store Analysis**: Processing user reviews and feedback
- **Social Media Mining**: Extracting topics from social media posts
- **Document Clustering**: Organizing large document collections by topics
- **Content Analysis**: Understanding thematic patterns in textual data

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT DATA                            │
│                    (CSV/Excel Files)                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING                           │
│  ┌─────────────────┐              ┌─────────────────┐          │
│  │   Turkish Path  │              │  English Path   │          │
│  │  - Text Cleaning│              │ - Dictionary    │          │
│  │  - Tokenization │              │   Creation      │          │
│  │  - BPE/WordPiece│              │ - Lemmatization │          │
│  └─────────────────┘              └─────────────────┘          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TF-IDF VECTORIZATION                         │
│            Convert text to numerical matrix                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MATRIX FACTORIZATION                            │
│    NMF: W (Document-Topic) × H (Topic-Word)                     │
│    NMTF: W (Document-Topic) × S (Topic-Topic) × H (Topic-Word)  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TOPIC ANALYSIS                               │
│  - Topic Word Extraction    - Document Classification           │
│  - Coherence Scoring        - Representative Documents          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT GENERATION                            │
│  - Word Clouds          - Excel Reports                         │
│  - Distribution Plots   - JSON Data                             │
│  - Database Storage     - Coherence Metrics                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Main Functions Documentation

### 1. process_turkish_file()

Processes Turkish text data through the complete preprocessing pipeline.

```python
def process_turkish_file(df, desired_columns: str, tokenizer=None, tokenizer_type=None):
    """
    Process Turkish text data with specialized Turkish NLP pipeline.
    
    Args:
        df (pandas.DataFrame): Input dataframe containing Turkish text
        desired_columns (str): Column name containing text to analyze
        tokenizer (optional): Pre-trained tokenizer object
        tokenizer_type (str): Either "bpe" or "wordpiece"
    
    Returns:
        tuple: (tdm, sozluk, sayisal_veri, tokenizer)
            - tdm: TF-IDF document-term matrix
            - sozluk: Vocabulary list
            - sayisal_veri: Numerical representation of documents
            - tokenizer: Trained tokenizer object
    """
```

**Process Flow:**
1. **Text Cleaning** (`clean_text_turkish`): Removes noise, special characters, and normalizes text
2. **Tokenizer Initialization**: Creates BPE or WordPiece tokenizer if not provided
3. **Tokenizer Training**: Trains tokenizer on the corpus to build vocabulary
4. **Vocabulary Extraction**: Extracts word tokens from trained tokenizer
5. **Numerical Conversion**: Converts text to numerical vectors using tokenizer
6. **TF-IDF Generation**: Creates weighted document-term matrix

### 2. process_english_file()

Handles English text processing with lemmatization and dictionary-based approaches.

```python
def process_english_file(df, desired_columns: str, lemmatize: bool):
    """
    Process English text data with English NLP pipeline.
    
    Args:
        df (pandas.DataFrame): Input dataframe containing English text
        desired_columns (str): Column name containing text to analyze
        lemmatize (bool): Whether to apply lemmatization
    
    Returns:
        tuple: (tdm, sozluk, sayisal_veri)
            - tdm: TF-IDF document-term matrix (same as sayisal_veri)
            - sozluk: Dictionary/vocabulary of terms
            - sayisal_veri: TF-IDF weighted document-term matrix
    """
```

**Process Flow:**
1. **Dictionary Creation** (`create_english_vocab`): Builds vocabulary with optional lemmatization
2. **TF-IDF Calculation** (`tf_idf_english`): Computes TF-IDF weights directly
3. **Matrix Preparation**: Prepares the final document-term matrix

### 3. process_file()

The main orchestration function that coordinates the entire topic modeling pipeline.

```python
def process_file(
    filepath: str,
    table_name: str,
    desired_columns: str,
    desired_topic_count: int,
    LEMMATIZE: bool,
    words_per_topic: int,
    tokenizer=None,
    LANGUAGE="TR",
    tokenizer_type="bpe",
    gen_topic_distribution=True,
    separator=",",
    nmf_method="nmf"
) -> dict:
    """
    Complete topic modeling pipeline from file input to results.
    
    Args:
        filepath (str): Path to input CSV/Excel file
        table_name (str): Unique identifier for this analysis run
        desired_columns (str): Column name containing text data
        desired_topic_count (int): Number of topics to extract
        LEMMATIZE (bool): Enable lemmatization (mainly for English)
        words_per_topic (int): Number of top words per topic to display
        tokenizer (optional): Pre-initialized tokenizer
        LANGUAGE (str): "TR" for Turkish, "EN" for English
        tokenizer_type (str): "bpe" or "wordpiece" for Turkish
        gen_topic_distribution (bool): Generate topic distribution plots
        separator (str): CSV separator character
        nmf_method (str): "nmf", "pnmf", or "nmtf" algorithm choice
    
    Returns:
        dict: Results containing:
            - state: "SUCCESS" or "FAILURE"
            - message: Status message
            - data_name: Analysis identifier
            - topic_word_scores: Topic-word associations
    """
```

**Complete Process Flow:**

1. **Environment Setup**
   - Creates necessary directories (`instance/`, `Output/`, `tfidf/`)
   - Initializes SQLite databases for topics and main data

2. **Data Loading & Cleaning**
   ```python
   # Special handling for CSV files
   if filepath.endswith(".csv"):
       # Replace problematic characters
       data = data.replace("|", ";")
       data = data.replace("\\t", "")
       data = data.replace("\\x00", "")
   ```
   - Loads CSV/Excel files with encoding handling
   - Filters data (e.g., country-specific filtering)
   - Removes duplicates and null values

3. **Database Storage**
   - Stores processed data in SQLite database
   - Creates table with specified `table_name`

4. **Language-Specific Processing**
   ```python
   if LANGUAGE == "TR":
       tdm, sozluk, sayisal_veri, tokenizer = process_turkish_file(...)
   elif LANGUAGE == "EN":
       tdm, sozluk, sayisal_veri = process_english_file(...)
   ```

5. **NMF Decomposition**
   ```python
   W, H = run_nmf(
       num_of_topics=int(desired_topic_count),
       sparse_matrix=tdm,
       norm_thresh=0.005,
       nmf_method=nmf_method
   )
   ```

6. **Topic Analysis & Output Generation**
   - Extracts dominant words and documents for each topic
   - Calculates coherence scores
   - Generates word clouds and distribution plots
   - Exports results to Excel format

### 4. run_topic_analysis()

The main entry point function that provides a simplified interface.

```python
def run_topic_analysis(
    filepath: str,
    column: str,
    separator: str = ",",
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
    filter_app: bool = False,
    data_filter_options: dict = None,
    emoji_map: bool = False,
    output_name: str = None,
    save_to_db: bool = False,
    output_dir: str = None
) -> dict:
    """
    Comprehensive topic modeling analysis using Non-negative Matrix Factorization (NMF).
    
    Args:
        filepath (str): Path to input CSV or Excel file
        column (str): Name of column containing text data
        separator (str): CSV file separator (default: ",")
        language (str): "TR" for Turkish, "EN" for English (default: "EN")
        topic_count (int): Number of topics to extract (default: 5)
        nmf_method (str): "nmf", "pnmf", or "nmtf" algorithm variant (default: "nmf")
        lemmatize (bool): Apply lemmatization for English (default: False)
        tokenizer_type (str): "bpe" or "wordpiece" for Turkish (default: "bpe")
        words_per_topic (int): Top words to show per topic (default: 15)
        word_pairs_out (bool): Create word pairs output (default: True)
        generate_wordclouds (bool): Create word cloud visualizations (default: True)
        export_excel (bool): Export results to Excel (default: True)
        topic_distribution (bool): Generate distribution plots (default: True)
        filter_app (bool): Enable app filtering (default: False)
        data_filter_options (dict): Advanced filtering options
        emoji_map (bool): Enable emoji processing (default: False)
        output_name (str): Custom output directory name (default: auto-generated)
        save_to_db (bool): Whether to persist data to database (default: False)
        output_dir (str): Base directory for outputs (default: current working directory)
    
    Returns:
        dict: Results containing:
            - state: "SUCCESS" or "FAILURE"
            - message: Status message
            - data_name: Analysis identifier
            - topic_word_scores: Topic-word associations
            - topic_doc_scores: Topic-document associations
            - coherence_scores: Coherence metrics
            - topic_relationships: S matrix (only for NMTF)
    """
```

---

## Workflow Process

### Detailed Step-by-Step Process

#### Phase 1: Data Preparation
```
Input File → Data Loading → Cleaning → Deduplication → Optional Database Storage
```

#### Phase 2: Text Processing (Language-Dependent)

**Turkish Path:**
```
Raw Text → Text Cleaning → Tokenizer Training → Vocabulary Building → 
Numerical Conversion → TF-IDF Matrix
```

**English Path:**
```
Raw Text → Dictionary Creation → Lemmatization (optional) → 
TF-IDF Calculation → Matrix Preparation
```

#### Phase 3: Topic Modeling
```
TF-IDF Matrix → NMF/NMTF Decomposition → W Matrix (Document-Topic) + 
H Matrix (Topic-Word) [+ S Matrix (Topic-Topic) for NMTF] → Topic Analysis
```

#### Phase 4: Results & Visualization
```
Topic Matrices → Word Extraction → Coherence Calculation → 
Word Clouds → Distribution Plots → Excel Export → JSON Storage
```

---

## Module Structure

### _functions/ Directory

#### common_language/
- **`emoji_processor.py`**: Emoji handling utilities
- **`topic_extractor.py`**: Cross-language topic analysis and extraction

#### english/
- **`english_entry.py`**: English text processing entry point
- **`english_preprocessor.py`**: Text cleaning and preprocessing
- **`english_vocabulary.py`**: Vocabulary creation
- **`english_text_encoder.py`**: Text-to-numerical conversion
- **`english_topic_analyzer.py`**: Topic extraction utilities
- **`english_topic_output.py`**: Topic visualization and output
- **`english_nmf_core.py`**: NMF implementation for English

#### turkish/
- **`turkish_entry.py`**: Turkish text processing entry point
- **`turkish_preprocessor.py`**: Turkish text cleaning and normalization
- **`turkish_tokenizer_factory.py`**: Tokenizer creation and training (BPE/WordPiece)
- **`turkish_text_encoder.py`**: Turkish text numerical conversion
- **`turkish_tfidf_generator.py`**: TF-IDF generation for Turkish

#### nmf/
- **`nmf_orchestrator.py`**: Main NMF orchestration function
- **`nmf_basic.py`**: Standard NMF implementation
- **`nmf_projective_basic.py`**: Basic projective NMF
- **`nmf_projective_enhanced.py`**: Enhanced projective NMF
- **`nmf_initialization.py`**: Matrix initialization strategies
- **`nmtf/`**: Non-negative Matrix Tri-Factorization implementation
  - **`nmtf.py`**: Core NMTF algorithm with topic relationships
  - **`nmtf_init.py`**: NMTF initialization utilities
  - **`nmtf_util.py`**: NMTF helper functions
  - **`extract_nmtf_topics.py`**: Topic extraction for NMTF results
  - **`example_usage.py`**: NMTF usage examples
- **`other/`**: Additional NMF algorithm variants
  - **`hierarchical_nmf.py`**: Hierarchical NMF implementation
  - **`nmf_onlineNMF.py`**: Online NMF for streaming data
  - **`symmetric_nmf.py`**: Symmetric NMF variant

#### tfidf/
- **`tfidf_english_calculator.py`**: English TF-IDF calculation
- **`tfidf_turkish_calculator.py`**: Turkish TF-IDF generation
- **`tfidf_tf_functions.py`**: Term frequency calculation functions
- **`tfidf_idf_functions.py`**: Inverse document frequency functions
- **`tfidf_bm25_turkish.py`**: BM25 implementation for Turkish

### utils/ Directory

#### analysis/
- **`coherence_score.py`**: Topic coherence evaluation
- **`distance_two_words.py`**: Word distance calculation
- **`umass_test.py`**: UMass coherence testing
- **`word_cooccurrence.py`**: Word co-occurrence analysis
- **`word_cooccurrence_analyzer.py`**: Advanced word co-occurrence analysis

#### console/
- **`console_manager.py`**: Console and logging management utilities

#### database/
- **`database_manager.py`**: Database connection and management utilities
- **`save_topics_db.py`**: Topic database saving utilities

#### export/
- **`export_excel.py`**: Excel report generation
- **`json_to_excel.py`**: JSON to Excel conversion utilities
- **`save_doc_score_pair.py`**: Document-topic score persistence
- **`save_word_score_pair.py`**: Word-score pair saving utilities

#### preprocess/
- **`combine_number_suffix.py`**: Number and suffix combination utilities

#### visualization/
- **`gen_cloud.py`**: Word cloud generation
- **`image_to_base.py`**: Image to base64 conversion
- **`topic_dist.py`**: Topic distribution visualization
- **`visualizer.py`**: General visualization utilities

#### agent/
- **`claude_prompt_generator.py`**: Claude AI prompt generation utilities
- **`claude_prompt_generator.html`**: HTML interface for prompt generation

#### other/
- **`backup.py`**: Backup and restoration utilities
- **`hierarchy_nmf.py`**: Hierarchical NMF utilities
- **`redis_bridge.py`**: Redis database bridge utilities
- Additional utility functions and experimental features


---

## Configuration Options

### Core Parameters

| Parameter | Type | Description                         | Default | Options |
|-----------|------|-------------------------------------|---------|---------|
| `language` | str | Text language                       | "EN" | "TR", "EN" |
| `topic_count` | int | Number of topics to extract         | 5 | 2-50+ |
| `words_per_topic` | int | Words per topic to display          | 15 | 5-30 |
| `lemmatize` | bool | Enable lemmatization (English)      | False | True, False |
| `tokenizer_type` | str | Tokenizer type (Turkish)            | "bpe" | "bpe", "wordpiece" |
| `nmf_method` | str | NMF Factorization algorithm         | "nmf" | "nmf", "pnmf", "nmtf" |
| `emoji_map` | bool | Enable emoji processing and mapping | False | True, False |
| `separator` | str | CSV file separator                  | "," | ",", ";", "\\t" |
| `filter_app` | bool | Enable data filtering               | False | True, False |
| `word_pairs_out` | bool | Create word pairs output            | True | True, False |
| `generate_wordclouds` | bool | Generate word clouds               | True | True, False |
| `export_excel` | bool | Export to Excel                    | True | True, False |
| `topic_distribution` | bool | Generate distribution plots        | True | True, False |
| `save_to_db` | bool | Save to database                   | False | True, False |
| `output_name` | str | Custom output name                  | None | Any string |
| `output_dir` | str | Base output directory              | None | Any path |

### Advanced Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `topic_distribution` | bool | Generate distribution plots | True |
| `generate_wordclouds` | bool | Generate word clouds | True |
| `export_excel` | bool | Export to Excel | True |
| `word_pairs_out` | bool | Calculate word co-occurrence | True |
| `save_to_db` | bool | Save to database | False |
| `output_name` | str | Custom output name | None |
| `output_dir` | str | Base output directory | None |
| `norm_thresh` | float | NMF normalization threshold | 0.005 |

### Data Filtering Options

When `filter_app` is enabled, you can configure advanced filtering using the `data_filter_options` dictionary:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `filter_app_name` | str | App name to filter by | "" |
| `filter_app_column` | str | Column name for app filtering | "PACKAGE_NAME" |
| `filter_app_country` | str | Country code to filter by (case-insensitive) | "" |
| `filter_app_country_column` | str | Column name for country filtering | "COUNTRY" |

**Example filtering configuration:**
```python
"filter_app": True,
"data_filter_options": {
    "filter_app_name": "com.example.app",      # Filter by specific app ID
    "filter_app_column": "PACKAGE_NAME",       # Column containing app identifiers
    "filter_app_country": "TR",                # Filter by Turkish data (case-insensitive)
    "filter_app_country_column": "COUNTRY"      # Column containing country codes
}
```

### Example Configuration

```python
# Example configuration using current API
result = run_topic_analysis(
    filepath="data/example.csv",
    column="text_content",
    language="TR",
    topic_count=8,
    words_per_topic=15,
    tokenizer_type="bpe",
    nmf_method="pnmf",
    lemmatize=True,
    separator=";",
    emoji_map=True,
    generate_wordclouds=True,
    export_excel=True,
    word_pairs_out=False,
    topic_distribution=True,
    filter_app=True,
    data_filter_options={
        "filter_app_name": "MyApp",
        "filter_app_column": "APP_NAME",
        "filter_app_country": "TR",
        "filter_app_country_column": "COUNTRY_CODE"
    },
    output_name="custom_analysis"
)
```

---

## Output Files

### Directory Structure
```
Output/
└── {output_name}/
    ├── {output_name}_topics.xlsx              # Topic-word matrix
    ├── {output_name}_coherence_scores.json    # Coherence metrics
    ├── {output_name}_document_dist.png        # Topic distribution plot
    ├── {output_name}_wordcloud_scores.json    # Word cloud data
    ├── {output_name}_top_docs.json            # Representative documents
    └── wordclouds/
        ├── Konu 00.png                       # Word cloud for topic 0
        ├── Konu 01.png                       # Word cloud for topic 1
        └── ...
```

### File Descriptions

#### Excel Report (`{output_name}_topics.xlsx`)
- **Topic Sheets**: Each topic gets its own worksheet
- **Word Scores**: Top words with their importance scores
- **Document References**: Representative documents for each topic

#### Coherence Scores (`{output_name}_coherence_scores.json`)
```json
{
    "gensim": {
       "umass_average": -1.4328882390292266,
        "umass_per_topic": {
            "topic_0": -1.4328882390292266,
            "topic_1": -1.1234567890123456,
            "topic_2": -0.9876543210987654
        }
    }
}
```

#### Word Cloud Data (`{output_name}_wordcloud_scores.json`)
```json
{
    "topic_0": {
        "word1": 0.15,
        "word2": 0.12,
        "word3": 0.10
    }
}
```

#### Representative Documents (`{output_name}_top_docs.json`)
```json
{
    "topic_0": [
        {
            "document": "Sample document text...",
            "score": 0.78
        }
    ]
}
```

---

## Usage Examples

### Example 1: Turkish App Store Reviews

```python
# Turkish app store reviews analysis
result = run_topic_analysis(
    filepath="data/app_reviews.csv",
    column="review_text",
    language="TR",                         # Turkish language
    topic_count=6,                        # Extract 6 topics
    words_per_topic=20,                   # 20 words per topic
    tokenizer_type="bpe",                 # Use BPE tokenization
    nmf_method="nmf",                     # Standard NMF
    lemmatize=False,                      # Not needed for Turkish
    separator=",",                        # CSV separator
    generate_wordclouds=True,             # Generate word clouds
    export_excel=True,                    # Export to Excel
    topic_distribution=True,              # Create distribution plots
    output_name="app_reviews_analysis"    # Custom output name
)

print(f"Analysis completed: {result['state']}")
print(f"Topics extracted: {len(result['topic_word_scores'])}")
```

### Example 2: English Academic Papers

```python
# English research papers analysis
result = run_topic_analysis(
    filepath="data/research_papers.csv",
    column="abstract",
    language="EN",                         # English language
    topic_count=10,                       # Extract 10 topics
    words_per_topic=25,                   # 25 words per topic
    nmf_method="pnmf",                    # Orthogonal NMF
    lemmatize=True,                       # Enable lemmatization
    separator=",",
    generate_wordclouds=True,
    export_excel=True,
    topic_distribution=True,
    output_name="research_analysis"       # Custom output name
)
```

### Example 3: Advanced Filtering Configuration

```python
# Advanced filtering configuration for app store reviews
result = run_topic_analysis(
    filepath="app_reviews.csv",
    column="review_text",
    language="TR",                         # Turkish language
    topic_count=5,                        # Extract 5 topics
    words_per_topic=15,                   # 15 words per topic
    tokenizer_type="bpe",                 # Use BPE tokenization
    nmf_method="nmf",                     # Standard NMF
    lemmatize=False,                      # Not needed for Turkish
    separator="|",                        # Pipe separator
    generate_wordclouds=True,             # Generate word clouds
    export_excel=True,                    # Export to Excel
    topic_distribution=True,              # Create distribution plots
    filter_app=True,                      # Enable filtering
    data_filter_options={
        "filter_app_name": "com.example.app",      # Filter by specific app
        "filter_app_column": "PACKAGE_NAME",       # Column containing app names
        "filter_app_country": "TR",                # Filter by country (case-insensitive)
        "filter_app_country_column": "COUNTRY"      # Column containing country codes
    },
    output_name="filtered_app_analysis"   # Custom output name
)

print(f"Filtered analysis result: {result['state']}")
```

### Example 4: Batch Processing Multiple Files

```python
files_to_process = [
    {
        "filepath": "data/dataset1.csv",
        "output_name": "analysis_1",
        "column": "text_content",
        "topic_count": 8
    },
    {
        "filepath": "data/dataset2.csv", 
        "output_name": "analysis_2",
        "column": "description",
        "topic_count": 12
    }
]

results = []
for file_config in files_to_process:
    result = run_topic_analysis(
        filepath=file_config["filepath"],
        column=file_config["column"],
        language="TR",
        topic_count=file_config["topic_count"],
        words_per_topic=15,
        tokenizer_type="bpe",
        nmf_method="nmf",
        lemmatize=True,
        separator=",",
        generate_wordclouds=True,
        export_excel=True,
        topic_distribution=True,
        output_name=file_config["output_name"]
    )
    results.append(result)

print(f"Processed {len(results)} files successfully")
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. File Encoding Issues

**Problem**: `UnicodeDecodeError` when reading CSV files

**Solution**:
```python
# The system handles this automatically, but if you encounter issues:
# - Ensure your CSV file is UTF-8 encoded
# - Check for special characters in your data
# - Use a text editor to convert file encoding to UTF-8
```

#### 2. Memory Issues with Large Datasets

**Problem**: `MemoryError` or system slowdown

**Solution**:
- Reduce the number of topics (`topic_count`)
- Filter your dataset to smaller chunks
- Use fewer words per topic (`words_per_topic`)
- Consider using a more powerful machine

#### 3. No Topics Generated

**Problem**: Empty or meaningless topics

**Solutions**:
- Check if your text column contains meaningful content
- Increase the minimum document length
- Adjust the `desired_topic_count` parameter 
- Ensure proper text cleaning for your language

#### 4. Tokenizer Training Fails

**Problem**: Tokenizer fails to train on Turkish text

**Solutions**:
- Ensure sufficient text data (minimum 1000 documents recommended)
- Check text quality and remove excessive noise
- Try switching between "bpe" and "wordpiece" tokenizers
- Verify that the text is actually in Turkish

#### 5. Poor Topic Quality

**Problem**: Topics contain mostly stopwords or irrelevant terms

**Solutions**:
- Improve text preprocessing
- Add domain-specific stopwords
- Adjust NMF parameters (`norm_thresh`)
- Try different `nmf_method` ("nmf" vs "pnmf" vs "nmtf")

#### 6. Database Lock Errors

**Problem**: `database is locked` error

**Solution**:
```python
# Delete existing database files if safe to do so:
# rm instance/topics.db
# rm instance/scopus.db
# Or use a different output_name for each run
```

#### 7. Data Filtering Issues

**Problem**: Filtering not working or "filter columns are not present" error

**Solutions**:
- Verify that the specified filter columns exist in your dataset
- Check column names for exact matches (case-sensitive)  
- Use `pandas.read_csv()` to inspect your data structure first
- Note that country filtering is now case-insensitive (automatically converted to uppercase)
- Ensure filter values match the data format (e.g., country codes should be uppercase like "TR", "US")

```python
# Debug filtering issues
import pandas as pd
df = pd.read_csv("your_file.csv")
print(df.columns.tolist())  # Check available columns
print(df['COUNTRY'].unique())  # Check available country values
```

#### 8. Missing Dependencies

**Problem**: Import errors for required packages

**Solution**:
```bash
# Install using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

#### 9. NMTF Convergence Issues

**Problem**: NMTF takes too long to converge or doesn't converge

**Solutions**:
- Reduce the number of topics (`topic_count`)
- Adjust the normalization threshold (`norm_thresh`) to a higher value (e.g., 0.01)
- Increase the convergence threshold (`epsilon`) for faster convergence
- Check that your data has sufficient complexity for the requested number of topics

```python
# Example configuration for faster NMTF convergence
result = run_topic_analysis(
    filepath="data/example.csv",
    column="text_content",
    language="TR",
    nmf_method="nmtf",
    topic_count=5,              # Reduce topic count
    words_per_topic=15,
    generate_wordclouds=True,
    export_excel=True,
    output_name="nmtf_analysis"
    # Note: norm_thresh is an internal parameter not exposed in the API
)
```

#### 10. NMTF Memory Issues

**Problem**: Out of memory errors when running NMTF

**Solutions**:
- NMTF requires more memory than standard NMF due to the additional S matrix
- Reduce the dataset size or filter to fewer documents
- Reduce the number of topics
- Consider using standard NMF if memory is limited
- Ensure you have sufficient RAM (recommended: 8GB+ for medium datasets)

### Performance Optimization Tips

1. **For Large Datasets**:
   - Use `gen_topic_distribution=False` to skip plotting
   - Set `gen_cloud=False` to skip word cloud generation
   - Reduce `words_per_topic` to 10-15

2. **For Better Topic Quality**:
   - Increase `desired_topic_count` for more granular topics
   - Use `nmf_method="pnmf"` for better topic separation
   - Enable `LEMMATIZE=True` for English text

3. **For Faster Processing**:
   - Use `tokenizer_type="bpe"` for Turkish (generally faster)
   - Pre-filter your data to remove very short documents
   - Use SSD storage for better I/O performance

### Getting Help

If you encounter issues not covered here:

1. Check the console output for detailed error messages
2. Verify your input data format and content
3. Ensure all dependencies are properly installed
4. Review the configuration parameters for your use case

---

*This documentation covers the complete MANTA application system. For additional support or feature requests, please refer to the project repository.*