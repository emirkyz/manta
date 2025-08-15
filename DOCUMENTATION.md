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
    N_TOPICS: int,
    tokenizer=None,
    LANGUAGE="TR",
    tokenizer_type="bpe",
    gen_topic_distribution=True,
    separator=",",
    nmf_type="nmf"
) -> dict:
    """
    Complete topic modeling pipeline from file input to results.
    
    Args:
        filepath (str): Path to input CSV/Excel file
        table_name (str): Unique identifier for this analysis run
        desired_columns (str): Column name containing text data
        desired_topic_count (int): Number of topics to extract
        LEMMATIZE (bool): Enable lemmatization (mainly for English)
        N_TOPICS (int): Number of top words per topic to display
        tokenizer (optional): Pre-initialized tokenizer
        LANGUAGE (str): "TR" for Turkish, "EN" for English
        tokenizer_type (str): "bpe" or "wordpiece" for Turkish
        gen_topic_distribution (bool): Generate topic distribution plots
        separator (str): CSV separator character
        nmf_type (str): "nmf" or "pnmf" algorithm choice
    
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
       nmf_method=nmf_type
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
def run_topic_analysis(filepath, table_name, desired_columns, options):
    """
    Simplified entry point for NMF/NMTF topic modeling.
    
    Args:
        filepath (str): Path to input file
        table_name (str): Analysis identifier
        desired_columns (str): Text column name
        options (dict): Configuration dictionary containing:
            - LEMMATIZE: bool
            - N_TOPICS: int (words per topic)
            - DESIRED_TOPIC_COUNT: int
            - tokenizer_type: str
            - nmf_type: str ("nmf", "pnmf", or "nmtf")
            - LANGUAGE: str
            - separator: str
            - gen_topic_distribution: bool
    
    Returns:
        dict: Process results with timing information and:
            - topic_relationships: S matrix (only for NMTF)
    """
```

---

## Workflow Process

### Detailed Step-by-Step Process

#### Phase 1: Data Preparation
```
Input File → Data Loading → Cleaning → Deduplication → Database Storage
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
TF-IDF Matrix → NMF Decomposition → W Matrix (Document-Topic) + 
H Matrix (Topic-Word) → Topic Analysis
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
| `LANGUAGE` | str | Text language                       | "TR" | "TR", "EN" |
| `desired_topic_count` | int | Number of topics to extract         | 5 | 2-50+ |
| `N_TOPICS` | int | Words per topic to display          | 15 | 5-30 |
| `LEMMATIZE` | bool | Enable lemmatization (English)      | True | True, False |
| `tokenizer_type` | str | Tokenizer type (Turkish)            | "bpe" | "bpe", "wordpiece" |
| `nmf_type` | str | NMF Factorization algorithm         | "nmf" | "nmf", "pnmf", "nmtf" |
| `emoji_map` | bool | Enable emoji processing and mapping | True | True, False |
| `separator` | str | CSV file separator                  | "," | ",", ";", "\\t" |
| `filter_app` | bool | Enable data filtering               | False | True, False |

### Advanced Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `gen_topic_distribution` | bool | Generate distribution plots | True |
| `gen_cloud` | bool | Generate word clouds | True |
| `save_excel` | bool | Export to Excel | True |
| `word_pairs_out` | bool | Calculate word co-occurrence | False |
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
options = {
    "LEMMATIZE": True,
    "N_TOPICS": 15,
    "DESIRED_TOPIC_COUNT": 8,
    "tokenizer_type": "bpe",
    "nmf_type": "pnmf",
    "LANGUAGE": "TR",
    "separator": ";",
    "emoji_map": True,
    "gen_cloud": True,
    "save_excel": True,
    "word_pairs_out": False,
    "gen_topic_distribution": True,
    "filter_app": True,
    "data_filter_options": {
        "filter_app_name": "MyApp",
        "filter_app_column": "APP_NAME",
        "filter_app_country": "TR",
        "filter_app_country_column": "COUNTRY_CODE"
    }
}
```

---

## Output Files

### Directory Structure
```
Output/
└── {table_name}/
    ├── {table_name}_topics.xlsx              # Topic-word matrix
    ├── {table_name}_coherence_scores.json    # Coherence metrics
    ├── {table_name}_document_dist.png        # Topic distribution plot
    ├── {table_name}_wordcloud_scores.json    # Word cloud data
    ├── {table_name}_top_docs.json            # Representative documents
    └── wordclouds/
        ├── Konu 00.png                       # Word cloud for topic 0
        ├── Konu 01.png                       # Word cloud for topic 1
        └── ...
```

### File Descriptions

#### Excel Report (`{table_name}_topics.xlsx`)
- **Topic Sheets**: Each topic gets its own worksheet
- **Word Scores**: Top words with their importance scores
- **Document References**: Representative documents for each topic

#### Coherence Scores (`{table_name}_coherence_scores.json`)
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

#### Word Cloud Data (`{table_name}_wordcloud_scores.json`)
```json
{
    "topic_0": {
        "word1": 0.15,
        "word2": 0.12,
        "word3": 0.10
    }
}
```

#### Representative Documents (`top_docs_{table_name}.json`)
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
# Configuration for Turkish app store reviews
options = {
    "LEMMATIZE": False,                    # Not needed for Turkish
    "N_TOPICS": 20,                       # 20 words per topic
    "DESIRED_TOPIC_COUNT": 6,             # Extract 6 topics
    "tokenizer_type": "bpe",              # Use BPE tokenization
    "nmf_type": "nmf",                    # Standard NMF
    "LANGUAGE": "TR",                     # Turkish language
    "separator": ",",                     # CSV separator
    "gen_cloud": True,                    # Generate word clouds
    "save_excel": True,                   # Export to Excel
    "gen_topic_distribution": True        # Create distribution plots
}

# Run the analysis
result = run_topic_analysis(
    filepath="data/app_reviews.csv",
    table_name="app_reviews_analysis",
    desired_columns="review_text",
    options=options
)

print(f"Analysis completed: {result['state']}")
print(f"Topics extracted: {len(result['topic_word_scores'])}")
```

### Example 2: English Academic Papers

```python
# Configuration for English research papers
options = {
    "LEMMATIZE": True,                     # Enable lemmatization
    "N_TOPICS": 25,                       # 25 words per topic
    "DESIRED_TOPIC_COUNT": 10,            # Extract 10 topics
    "tokenizer_type": None,               # Not used for English
    "nmf_type": "pnmf",                  # Orthogonal NMF
    "LANGUAGE": "EN",                     # English language
    "separator": ",",
    "gen_cloud": True,
    "save_excel": True,
    "gen_topic_distribution": True
}

# Run the analysis
result = run_topic_analysis(
    filepath="data/research_papers.csv",
    table_name="research_analysis",
    desired_columns="abstract",
    options=options
)
```

### Example 3: Advanced Filtering Configuration

```python
# Configuration for filtering app store reviews by specific app and country
options = {
    "LEMMATIZE": False,                    # Not needed for Turkish
    "N_TOPICS": 15,                       # 15 words per topic
    "DESIRED_TOPIC_COUNT": 5,             # Extract 5 topics
    "tokenizer_type": "bpe",              # Use BPE tokenization
    "nmf_type": "nmf",                    # Standard NMF
    "LANGUAGE": "TR",                     # Turkish language
    "separator": "|",                     # Pipe separator
    "gen_cloud": True,                    # Generate word clouds
    "save_excel": True,                   # Export to Excel
    "gen_topic_distribution": True,       # Create distribution plots
    "filter_app": True,                   # Enable filtering
    "data_filter_options": {
        "filter_app_name": "com.example.app",      # Filter by specific app
        "filter_app_column": "PACKAGE_NAME",       # Column containing app names
        "filter_app_country": "TR",                # Filter by country (case-insensitive)
        "filter_app_country_column": "COUNTRY"      # Column containing country codes
    }
}

# Run analysis with filtering
result = run_topic_analysis(
    filepath="app_reviews.csv",
    table_name="filtered_app_analysis",
    desired_columns="review_text",
    options=options
)

print(f"Filtered analysis result: {result['state']}")
```

### Example 4: Batch Processing Multiple Files

```python
files_to_process = [
    {
        "filepath": "data/dataset1.csv",
        "table_name": "analysis_1",
        "column": "text_content",
        "topics": 8
    },
    {
        "filepath": "data/dataset2.csv", 
        "table_name": "analysis_2",
        "column": "description",
        "topics": 12
    }
]

results = []
for file_config in files_to_process:
    options = {
        "LEMMATIZE": True,
        "N_TOPICS": 15,
        "DESIRED_TOPIC_COUNT": file_config["topics"],
        "tokenizer_type": "bpe",
        "nmf_type": "nmf",
        "LANGUAGE": "TR",
        "separator": ",",
        "gen_cloud": True,
        "save_excel": True,
        "gen_topic_distribution": True
    }
    
    result = run_topic_analysis(
        filepath=file_config["filepath"],
        table_name=file_config["table_name"],
        desired_columns=file_config["column"],
        options=options
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
- Reduce the number of topics (`DESIRED_TOPIC_COUNT`)
- Filter your dataset to smaller chunks
- Use fewer words per topic (`N_TOPICS`)
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
- Try different `nmf_type` ("nmf" vs "pnmf" vs "nmtf")

#### 6. Database Lock Errors

**Problem**: `database is locked` error

**Solution**:
```python
# Delete existing database files if safe to do so:
# rm instance/topics.db
# rm instance/scopus.db
# Or use a different table_name for each run
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
- Reduce the number of topics (`DESIRED_TOPIC_COUNT`)
- Adjust the normalization threshold (`norm_thresh`) to a higher value (e.g., 0.01)
- Increase the convergence threshold (`epsilon`) for faster convergence
- Check that your data has sufficient complexity for the requested number of topics

```python
# Example configuration for faster NMTF convergence
options = {
    "nmf_type": "nmtf",
    "DESIRED_TOPIC_COUNT": 5,  # Reduce topic count
    "norm_thresh": 0.01,       # Higher threshold for faster convergence
    # ... other options
}
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
   - Reduce `N_TOPICS` to 10-15

2. **For Better Topic Quality**:
   - Increase `desired_topic_count` for more granular topics
   - Use `nmf_type="pnmf"` for better topic separation
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