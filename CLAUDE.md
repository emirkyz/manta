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
```

### Project Structure and Architecture

This is a standalone NMF (Non-negative Matrix Factorization) topic modeling system that processes text data in both Turkish and English languages.

#### Core Architecture
- **Main Pipeline**: `standalone_nmf.py` orchestrates the entire process
- **Language Processing**: Separate modules for Turkish (`functions/turkish/`) and English (`functions/english/`) text processing
- **NMF Algorithms**: Multiple NMF implementations in `functions/nmf/` including basic NMF and OPNMF
- **TF-IDF Processing**: Language-specific TF-IDF implementations in `functions/tfidf/`
- **Utilities**: Helper functions in `utils/` for visualization, export, and analysis

#### Key Components
- **Text Preprocessing**: 
  - Turkish: Uses BPE/WordPiece tokenizers with custom cleaning (`functions/turkish/temizle.py`)
  - English: Traditional preprocessing with lemmatization options (`functions/english/process.py`)
- **Database Integration**: SQLite databases stored in `instance/` for topics and main data
- **Output Generation**: Word clouds, Excel exports, coherence scores, and topic distributions in `Output/`

#### Data Flow
1. CSV/Excel file → Data cleaning and preprocessing
2. Tokenization (language-specific)
3. TF-IDF matrix generation
4. NMF decomposition (W and H matrices)
5. Topic analysis and extraction
6. Visualization and export (word clouds, Excel, plots)

#### Configuration
The main script uses an options dictionary for configuration:
- `LANGUAGE`: "TR" for Turkish, "EN" for English
- `tokenizer_type`: "bpe" or "wordpiece" (Turkish only)
- `nmf_type`: "nmf" or "opnmf"
- `DESIRED_TOPIC_COUNT`: Number of topics to extract
- `N_TOPICS`: Number of words per topic

#### Output Structure
Results are saved to `Output/{table_name}/` containing:
- Excel files with topic-word scores
- Word cloud images for each topic
- Topic distribution plots
- Coherence scores (JSON)
- Top documents per topic (JSON)