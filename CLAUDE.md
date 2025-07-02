# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation
```bash
# Install dependencies using uv (recommended)
uv pip install -r requirements.txt

# Alternative with pip
pip install -r requirements.txt
```

### Running the Application
```bash
# Run the main topic modeling script
python standalone_nmf.py

# Run coherence evaluation for model selection
python coherence_eval_nmf.py

# Build documentation
python build_docs.py
```

### Testing
No formal test suite exists. The project includes evaluation scripts:
- `coherence_eval_nmf.py` - For model coherence evaluation
- Utility scripts in `utils/` for testing specific functionality  

## Project Structure and Architecture

This is a standalone NMF (Non-negative Matrix Factorization) topic modeling system that processes text data in both Turkish and English languages.

### Core Architecture
The system follows a modular architecture with clear separation of concerns:

- **Main Pipeline**: `standalone_nmf.py` orchestrates the entire process
- **Language Processing**: Separate modules for Turkish (`functions/turkish/`) and English (`functions/english/`) text processing
- **NMF Algorithms**: Multiple NMF implementations in `functions/nmf/` including standard NMF and projective variants (OPNMF)
- **TF-IDF Processing**: Language-specific TF-IDF implementations in `functions/tfidf/`
- **Utilities**: Helper functions in `utils/` for visualization, export, and analysis
- **Common Language**: Shared functionality in `functions/common_language/` for cross-language processing

### Key Components

#### Text Preprocessing
- **Turkish**: Uses modern tokenization (BPE/WordPiece) with emoji processing and custom text cleaning
  - `turkish_preprocessor.py` - Text cleaning and normalization
  - `turkish_tokenizer_factory.py` - Tokenizer initialization and training
  - `turkish_text_encoder.py` - Text-to-numerical conversion
- **English**: Traditional preprocessing with optional lemmatization
  - `english_preprocessor.py` - Text cleaning and preprocessing
  - `english_vocabulary.py` - Vocabulary creation and management
  - `english_text_encoder.py` - Text vectorization

#### NMF Implementation
- `nmf_orchestrator.py` - Main NMF interface and coordinator
- `nmf_multiplicative_updates.py` - Standard NMF algorithm
- `nmf_projective_basic.py` - Basic projective NMF (OPNMF)
- `nmf_projective_enhanced.py` - Enhanced projective NMF
- `nmf_initialization.py` - Matrix initialization strategies

#### TF-IDF Processing
- Language-specific TF-IDF calculators with different weighting schemes
- BM25 implementation for Turkish texts
- Modular TF and IDF function implementations

### Data Flow
1. **Input Processing**: CSV/Excel file reading and column selection
2. **Language Detection**: Automatic routing to Turkish or English processing pipeline
3. **Text Preprocessing**: Language-specific cleaning, tokenization, and normalization
4. **Vectorization**: TF-IDF matrix generation with appropriate weighting
5. **Topic Modeling**: NMF decomposition producing W (document-topic) and H (topic-word) matrices
6. **Analysis**: Topic extraction, coherence calculation, and document analysis
7. **Output Generation**: Word clouds, Excel exports, visualizations, and database storage

### Configuration System
The main script uses a comprehensive options dictionary:

**Core Parameters:**
- `LANGUAGE`: "TR" for Turkish, "EN" for English
- `DESIRED_TOPIC_COUNT`: Number of topics to extract
- `N_TOPICS`: Number of top words per topic
- `nmf_type`: "nmf" (standard) or "opnmf" (projective)

**Language-Specific:**
- `tokenizer_type`: "bpe" or "wordpiece" (Turkish only)
- `LEMMATIZE`: Enable lemmatization (English only)
- `emoji_map`: EmojiMap instance for Turkish emoji processing

**Output Control:**
- `gen_cloud`: Generate word cloud visualizations
- `save_excel`: Export results to Excel format
- `gen_topic_distribution`: Create topic distribution plots

### Database Integration
- SQLite databases stored in `instance/` directory
- `topics.db` - Topic modeling results and metadata
- `scopus.db` - Research data storage (if applicable)
- Automated database schema management through SQLAlchemy

### Output Structure
Results organized in `Output/{table_name}/` containing:
- Excel files with detailed topic-word scores and statistics
- Word cloud PNG images for each topic
- Topic distribution plots showing document-topic relationships
- JSON files with coherence scores and top documents per topic
- Database records for programmatic access

### File Naming Convention
The project follows a consistent naming pattern:
- **Language modules**: `{language}_{functionality}.py`
- **Algorithm modules**: `{algorithm}_{variant}.py`
- **Utility modules**: Descriptive names based on functionality
- **Output files**: `{table_name}_{algorithm}_{tokenizer}_{topic_count}` pattern