# NMF Standalone

This project performs topic modeling on text data using Non-negative Matrix Factorization (NMF). It supports both English and Turkish languages and can process `.csv` and `.xlsx` files. The main script, `standalone_nmf.py`, handles the entire pipeline from data preprocessing to topic extraction and visualization.

## Project Structure

```
nmf-standalone/
├── functions/
│   ├── english/
│   ├── nmf/
│   ├── tfidf/
│   └── turkish/
├── utils/
│   ├── other/
├── veri_setleri/
├── instance/
├── Output/
├── pyproject.toml
├── README.md
├── requirements.txt
├── standalone_nmf.py
└── uv.lock
```

-   **`functions/`**: Contains the core logic for the NMF pipeline, with separate modules for English and Turkish text processing, TF-IDF calculation, and NMF algorithms.
-   **`utils/`**: Includes helper functions for tasks like generating word clouds, calculating coherence scores, and exporting results.
-   **`veri_setleri/`**: Default directory for input datasets.
-   **`instance/`**: Stores databases created during the process (e.g., `topics.db`, `scopus.db`).
-   **`Output/`**: Directory where all the output files, such as topic reports, word clouds, and distribution plots, are saved.
-   **`standalone_nmf.py`**: The main executable script to run the topic modeling process.
-   **`requirements.txt`**: A list of Python packages required for the project.

## Installation

To run this project, it's recommended to use a virtual environment.

1.  **Create a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies using `uv`:**

    This project uses `uv` for fast dependency management. If you don't have `uv`, you can install it following the official instructions.

    ```bash
    uv pip install -r requirements.txt
    ```

## Usage

The main entry point for running the topic modeling is the `standalone_nmf.py` script. You can modify this script to set the parameters for your analysis.

The `run_standalone_nmf` function in the script is what you need to configure and run.

Here's an example of how you might call this function within the script:

```python
from functions.turkish.emoji_map import EmojiMap

if __name__ == "__main__":
    # Example for a Turkish dataset
    turkish_options = {
        "LEMMATIZE": False,
        "N_TOPICS": 15,
        "DESIRED_TOPIC_COUNT": 10,
        "tokenizer_type": "bpe",
        "tokenizer": None,
        "nmf_type": "nmf",
        "LANGUAGE": "TR",
        "separator": ",",
        "gen_cloud": True,
        "save_excel": True,
        "gen_topic_distribution": True,
        "filter_app": False,
        "filter_app_name": "",
        "emoji_map": EmojiMap()
    }
    
    run_standalone_nmf(
        filepath="veri_setleri/your_turkish_data.csv",
        table_name="my_turkish_analysis",
        desired_columns="text_column",
        options=turkish_options
    )

    # Example for an English dataset
    english_options = {
        "LEMMATIZE": True,
        "N_TOPICS": 20,
        "DESIRED_TOPIC_COUNT": 8,
        "tokenizer_type": None,
        "tokenizer": None,
        "nmf_type": "opnmf",
        "LANGUAGE": "EN",
        "separator": ",",
        "gen_cloud": True,
        "save_excel": True,
        "gen_topic_distribution": True,
        "filter_app": False,
        "filter_app_name": "",
        "emoji_map": None
    }
    
    run_standalone_nmf(
        filepath="veri_setleri/your_english_data.csv",
        table_name="my_english_analysis",
        desired_columns="text_column",
        options=english_options
    )

```

To run the script, simply execute it from your terminal:

```bash
python standalone_nmf.py
```

### Parameters

The `run_standalone_nmf` function takes the following parameters:

-   `filepath`: Path to your input `.csv` or `.xlsx` file.
-   `table_name`: A unique name for your analysis run. This is used for naming output files and database tables.
-   `desired_columns`: The name of the column in your data file that contains the text to be analyzed.
-   `options`: A dictionary containing all configuration options:

#### Options Dictionary Structure

**Core Parameters:**
-   `LANGUAGE`: `"TR"` for Turkish or `"EN"` for English.
-   `DESIRED_TOPIC_COUNT`: The number of topics to extract.
-   `N_TOPICS`: The number of top words to display for each topic.
-   `nmf_type`: The NMF algorithm to use (`"nmf"` or `"opnmf"`).

**Language-Specific Parameters:**
-   `LEMMATIZE`: Set to `True` for English text to enable lemmatization (ignored for Turkish).
-   `tokenizer_type`: For Turkish, choose between `"bpe"` (Byte-Pair Encoding) or `"wordpiece"`.
-   `tokenizer`: Pre-initialized tokenizer instance (optional, set to `None` for auto-initialization).
-   `emoji_map`: EmojiMap instance for Turkish emoji processing (use `EmojiMap()` for Turkish, `None` for English).

**File Processing Parameters:**
-   `separator`: The separator used in your `.csv` file (e.g., `,`, `;`).
-   `filter_app`: Set to `True` to filter data by application name.
-   `filter_app_name`: Application name to filter by (when `filter_app` is `True`).

**Output Generation Parameters:**
-   `gen_cloud`: Set to `True` to generate word cloud images for each topic.
-   `save_excel`: Set to `True` to export results to Excel format.
-   `gen_topic_distribution`: Set to `True` to generate topic distribution plots.

## Outputs

The script generates several outputs in the `Output/` directory, organized in a subdirectory named after your `table_name`:

-   **Topic-Word Excel File**: An `.xlsx` file containing the top words for each topic and their scores.
-   **Word Clouds**: PNG images of word clouds for each topic.
-   **Topic Distribution Plot**: A plot showing the distribution of documents across topics.
-   **Coherence Scores**: A JSON file with coherence scores for the topics.
-   **Top Documents**: A JSON file listing the most representative documents for each topic.
