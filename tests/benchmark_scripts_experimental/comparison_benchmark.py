"""
Fixed Comparison Benchmark Script - 5-Way Topic Modeling

Key fixes:
1. Proper input format handling for Gensim models
2. Consistent preprocessing pipeline for all models
3. Fixed coherence calculation
4. Better topic extraction methods
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from scipy import sparse

# --- New Imports for Extended Comparison ---
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.decomposition import LatentDirichletAllocation as SklearnLDA
from sklearn.feature_extraction.text import TfidfTransformer
import gensim
import gensim.corpora as corpora
from gensim.models import Nmf as GensimNMF_model
from gensim.models import LdaModel as GensimLDAModel
from gensim.models.coherencemodel import CoherenceModel

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import MANTA components
from manta._functions.common_language.emoji_processor import EmojiMap
from manta._functions.english.english_entry import process_english_file
from manta._functions.nmf import run_nmf
from manta._functions.turkish.turkish_entry import process_turkish_file
from manta._functions.turkish.turkish_tokenizer_factory import init_tokenizer
from manta.utils.database.database_manager import DatabaseManager
from manta.utils.console.console_manager import ConsoleManager


class ComparisonBenchmark:
    """
    Fixed benchmark class that ensures all models receive properly formatted input.
    """

    def __init__(self, console: Optional[ConsoleManager] = None):
        self.console = console or ConsoleManager()
        self.timing_results = {}
        self.preprocessing_data = {}

    def record_time(self, step_name: str, start_time: float):
        """Record timing for a processing step."""
        elapsed = time.time() - start_time
        self.timing_results[step_name] = elapsed
        self.console.print_status(f"{step_name} completed in {elapsed:.2f}s", "info")

    def load_and_preprocess_data(self, filepath: str, desired_columns: str, options: Dict[str, Any]) -> pd.DataFrame:
        """Load and preprocess data using MANTA's exact approach."""
        step_start = time.time()
        self.console.print_status("Step 1: Loading input file...", "processing")

        if str(filepath).endswith(".csv"):
            df = pd.read_csv(filepath, encoding="utf-8", sep=options["separator"], engine="python", on_bad_lines="skip")
        elif str(filepath).endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        self.record_time("Data Loading", step_start)

        step_start = time.time()
        self.console.print_status("Step 2: Preprocessing data...", "processing")
        if desired_columns not in df.columns:
            raise KeyError(f"Column '{desired_columns}' not found. Available columns: {', '.join(df.columns)}")

        df = df[[desired_columns]]
        initial_count = len(df)
        df = df.drop_duplicates().dropna()

        if len(df) == 0:
            raise ValueError("No data remaining after removing duplicates and null values")
        if len(df) < initial_count * 0.1:
            self.console.print_status(f"Warning: Only {len(df)} rows remain from original {initial_count}", "warning")

        self.console.print_status(f"Preprocessed dataset has {len(df)} rows", "info")
        self.record_time("Data Preprocessing", step_start)
        return df

    def perform_text_processing(self, df: pd.DataFrame, desired_columns: str, options: Dict[str, Any]):
        """Perform language-specific text processing and store results for all models."""
        step_start = time.time()
        self.console.print_status(f"Step 3: Starting text processing ({options['LANGUAGE']})...", "processing")

        if options["LANGUAGE"] == "TR":
            tdm_tfidf, vocab, counterized_data, text_array, options["tokenizer"], _ = process_turkish_file(
                df, desired_columns, options["tokenizer"], tokenizer_type=options["tokenizer_type"], emoji_map=None
            )
        elif options["LANGUAGE"] == "EN":
            tdm_tfidf, vocab, counterized_data, text_array, _ = process_english_file(
                df, desired_columns, options["LEMMATIZE"], emoji_map=None
            )
        else:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}")

        self.console.print_status("Text processing completed", "success")
        self.record_time("Text Processing", step_start)

        # Create properly tokenized texts for Gensim models
        texts_as_tokens = []
        if isinstance(text_array[0], str):
            # Split each document into tokens
            texts_as_tokens = [doc.split() for doc in text_array]
        else:
            # If already tokenized
            texts_as_tokens = text_array

        # Create Gensim dictionary for coherence calculation
        gensim_dictionary = corpora.Dictionary(texts_as_tokens)
        gensim_dictionary.filter_extremes(no_below=1, no_above=1.0)  # Keep all words

        # Convert vocab list to proper format if needed
        if isinstance(vocab, dict):
            vocab_list = [word for word, idx in sorted(vocab.items(), key=lambda x: x[1])]
        else:
            vocab_list = vocab

        self.preprocessing_data = {
            'tdm_tfidf': tdm_tfidf,  # TF-IDF matrix for NMF models
            'tdm_counts': counterized_data,  # Raw counts matrix for LDA models  
            'vocab': vocab_list,  # Vocabulary list
            'text_array': text_array,  # Original processed texts
            'texts_as_tokens': texts_as_tokens,  # Tokenized texts for Gensim
            'gensim_id2word': gensim_dictionary,  # Gensim dictionary for coherence
            'options': options.copy()
        }

    def _extract_topics_from_matrix(self, topic_word_matrix: np.ndarray, vocab: list, num_words: int) -> Dict:
        """Helper to extract topic words from a topic-word matrix (e.g., sklearn's components_)."""
        topic_word_scores = {}
        for topic_idx, topic in enumerate(topic_word_matrix):
            topic_name = f"Topic_{topic_idx + 1}"

            # FIX 2: Ensure we get the highest scoring words
            top_word_indices = topic.argsort()[-num_words:][::-1]  # Get top words in descending order
            word_scores = {}
            for i in top_word_indices:
                if i < len(vocab):  # Safety check
                    word_scores[vocab[i]] = float(topic[i])  # Convert to standard float

            topic_word_scores[topic_name] = word_scores
        return topic_word_scores

    def _create_gensim_inputs(self):
        """Create Gensim dictionary and corpus from preprocessed data."""
        if 'gensim_corpus' not in self.preprocessing_data:
            self.console.print_status("Creating Gensim dictionary and corpus...", "processing")

            texts_as_tokens = self.preprocessing_data['texts_as_tokens']
            
            # Use the existing dictionary but create a fresh one for corpus consistency
            id2word = corpora.Dictionary(texts_as_tokens)
            # More lenient filtering to maintain vocabulary consistency
            id2word.filter_extremes(no_below=1, no_above=0.95, keep_n=None)

            # Create corpus
            corpus = [id2word.doc2bow(text) for text in texts_as_tokens]

            self.preprocessing_data['gensim_corpus_id2word'] = id2word
            self.preprocessing_data['gensim_corpus'] = corpus

        return self.preprocessing_data['gensim_corpus_id2word'], self.preprocessing_data['gensim_corpus']

    def calculate_generic_coherence(self, topics: Dict, model_name: str) -> float:
        """Calculates coherence score for any model using a unified Gensim CoherenceModel."""
        try:
            self.console.print_status(f"Calculating coherence for {model_name}...", "processing")

            # Use the original tokenized texts for coherence calculation
            texts = self.preprocessing_data['texts_as_tokens']

            # Extract topic words (just the words, not the scores)
            topic_words = []
            for topic_dict in topics.values():
                words = list(topic_dict.keys())
                topic_words.append(words)

            # Use the existing Gensim dictionary from preprocessing
            gensim_dictionary = self.preprocessing_data['gensim_id2word']

            # Calculate coherence using c_v metric with proper Gensim dictionary
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=texts,
                dictionary=gensim_dictionary,
                coherence='c_v'
            )

            return coherence_model.get_coherence()

        except Exception as e:
            self.console.print_status(f"Coherence calculation failed for {model_name}: {e}", "warning")
            return 0.0

    def run_and_benchmark_model(self, model_name: str) -> Dict:
        """Runs a specific topic model, times it, and calculates its coherence."""
        step_start = time.time()
        self.console.print_status(f"Step 4: Starting analysis for -> {model_name}", "processing")

        options = self.preprocessing_data['options']
        vocab = self.preprocessing_data['vocab']
        num_topics = int(options["DESIRED_TOPIC_COUNT"])
        num_words = int(options["N_TOPICS"])
        topic_word_scores = {}

        try:
            if model_name == "Custom NMF":
                # Use TF-IDF matrix for NMF
                tdm_tfidf = self.preprocessing_data['tdm_tfidf']
                nmf_output = run_nmf(num_of_topics=num_topics, sparse_matrix=tdm_tfidf, nmf_method=options["nmf_type"])
                topic_word_scores = self._extract_topics_from_matrix(nmf_output["H"], vocab, num_words)

            elif model_name == "Scikit-learn NMF":
                # Use TF-IDF matrix for NMF
                tdm_tfidf = self.preprocessing_data['tdm_tfidf']
                if sparse.issparse(tdm_tfidf):
                    tdm_array = tdm_tfidf.toarray()
                else:
                    tdm_array = tdm_tfidf

                # Ensure non-negative values for NMF
                tdm_array = np.abs(tdm_array)

                model = SklearnNMF(
                    n_components=num_topics,
                    random_state=42,
                    init='nndsvd',  # Use standard initialization for TF-IDF data
                    max_iter=1000,  # Increase iterations
                    solver='cd',  # Coordinate descent solver works well with TF-IDF
                    alpha_W=0.01,  # Reduce regularization
                    alpha_H=0.01,
                    l1_ratio=0.0  # L2 regularization only
                )
                model.fit(tdm_array)
                topic_word_scores = self._extract_topics_from_matrix(model.components_, vocab, num_words)

            elif model_name == "Scikit-learn LDA":
                # Use TF-IDF matrix for LDA (matching the successful standalone_lda.py approach)
                tdm_tfidf = self.preprocessing_data['tdm_tfidf']
                if sparse.issparse(tdm_tfidf):
                    tdm_array = tdm_tfidf.toarray()
                else:
                    tdm_array = tdm_tfidf

                # Ensure non-negative values (TF-IDF should already be non-negative)
                tdm_array = np.abs(tdm_array)

                model = SklearnLDA(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=50,  # Match standalone_lda.py configuration
                    learning_method='online',  # Match standalone_lda.py configuration  
                    learning_offset=50.0,
                    doc_topic_prior=None,  # Auto-tune alpha
                    topic_word_prior=None,  # Auto-tune beta
                    n_jobs=1,  # Single-threaded for consistency
                    verbose=0
                )
                model.fit(tdm_array)
                topic_word_scores = self._extract_topics_from_matrix(model.components_, vocab, num_words)

            elif model_name in ["Gensim NMF", "Gensim LDA"]:
                # Use properly created Gensim inputs
                id2word, corpus = self._create_gensim_inputs()

                if model_name == "Gensim NMF":
                    # Apply TF-IDF transformation for NMF
                    tfidf = gensim.models.TfidfModel(corpus)
                    corpus_tfidf = tfidf[corpus]

                    model = GensimNMF_model(
                        corpus=corpus_tfidf,
                        num_topics=num_topics,
                        id2word=id2word,
                        random_state=42,
                        passes=50,  # More passes for better convergence
                        minimum_probability=0.001,  # Lower threshold for more words
                        normalize=True,
                        chunksize=100
                    )
                else:  # Gensim LDA
                    model = GensimLDAModel(
                        corpus=corpus,
                        num_topics=num_topics,
                        id2word=id2word,
                        random_state=42,
                        passes=50,  # More passes for better convergence
                        alpha='symmetric',  # Better for diverse topics
                        eta='auto',   # Auto-tune beta
                        minimum_probability=0.001,  # Lower threshold for more words
                        chunksize=100,
                        update_every=1,
                        iterations=100  # More iterations per pass
                    )

                # Extract topics with proper formatting
                topics = model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
                topic_word_scores = {}
                for topic_id, topic_words in topics:
                    topic_name = f"Topic_{topic_id + 1}"
                    word_scores = {}
                    for word, score in topic_words:
                        word_scores[word] = float(score)
                    topic_word_scores[topic_name] = word_scores

            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Calculate coherence
            coherence_score = self.calculate_generic_coherence(topic_word_scores, model_name)
            elapsed_time = time.time() - step_start
            self.timing_results[model_name] = elapsed_time

            self.console.print_status(
                f"{model_name} completed in {elapsed_time:.2f}s with Coherence C_V: {coherence_score:.4f}",
                "success"
            )

            return {
                "time": elapsed_time,
                "coherence": coherence_score,
                "topics": topic_word_scores
            }

        except Exception as e:
            elapsed_time = time.time() - step_start
            self.console.print_status(f"{model_name} failed: {e}", "error")
            return {
                "time": elapsed_time,
                "coherence": 0.0,
                "topics": {},
                "error": str(e)
            }

    def run_comparison_benchmark(self, filepath: str, table_name: str, desired_columns: str, options: Dict[str, Any], output_base_dir: Optional[str] = None):
        """Runs the complete 5-way comparison benchmark."""
        try:
            self.console.start_timing()
            self.console.print_header("5-Way Topic Modeling Comparison Benchmark (FIXED)")
            self.console.display_config(options, filepath, desired_columns, table_name)

            setup_start = time.time()
            db_config = DatabaseManager.initialize_database_config(output_base_dir)
            table_output_dir = db_config.output_dir / table_name
            table_output_dir.mkdir(parents=True, exist_ok=True)
            self.record_time("Setup", setup_start)

            init_start = time.time()
            if not options.get("tokenizer"):
                options["tokenizer"] = init_tokenizer(tokenizer_type=options["tokenizer_type"])
            options["emoji_map"] = EmojiMap() if options.get("emoji_map") else None
            self.record_time("Initialization", init_start)

            df = self.load_and_preprocess_data(filepath, desired_columns, options)
            self.perform_text_processing(df, desired_columns, options)

            models_to_run = ["Custom NMF", "Scikit-learn NMF", "Scikit-learn LDA", "Gensim NMF", "Gensim LDA"]
            benchmark_results = {}

            for model_name in models_to_run:
                result = self.run_and_benchmark_model(model_name)
                benchmark_results[model_name] = result

            total_time = self.console.get_total_time()
            self.print_comparison_summary(benchmark_results)

            # Save results
            all_topics_path = table_output_dir / "all_model_topics_fixed.json"
            def serialize_numpy(obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            with open(all_topics_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_results, f, ensure_ascii=False, indent=2, default=serialize_numpy)
            self.console.print_status(f"All topic results saved to {all_topics_path}", "info")

            return {
                "state": "SUCCESS",
                "benchmark_results": benchmark_results,
                "total_time": total_time,
                "output_directory": str(table_output_dir)
            }

        except Exception as e:
            import traceback
            self.console.print_status(f"Benchmark failed: {e}", "error")
            print(f"Full error trace: {traceback.format_exc()}")
            return {"state": "FAILURE", "message": str(e)}

    def print_comparison_summary(self, benchmark_results: Dict):
        """Prints a final comparison summary table."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY (FIXED VERSION)")
        print("="*80)

        summary_data = []
        for name, res in benchmark_results.items():
            summary_data.append({
                "Model": name,
                "Time (s)": res.get("time", float('inf')),
                "Coherence (C_v)": res.get("coherence", 0.0),
                "Status": "Failed" if "error" in res else "Success"
            })

        df = pd.DataFrame(summary_data)

        print("\n--- Results sorted by Speed (fastest first) ---")
        print(df.sort_values(by="Time (s)").to_string(index=False, float_format="%.4f"))

        print("\n--- Results sorted by Coherence (best first) ---")
        print(df.sort_values(by="Coherence (C_v)", ascending=False).to_string(index=False, float_format="%.4f"))

        # Filter out failed runs before determining winners
        successful_runs = df[df['Status'] == 'Success']
        if not successful_runs.empty:
            fastest = successful_runs.loc[successful_runs["Time (s)"].idxmin()]
            best_coherence = successful_runs.loc[successful_runs["Coherence (C_v)"].idxmax()]

            print("\n--- WINNERS ---")
            print(f"🚀 Fastest Model : {fastest['Model']} ({fastest['Time (s)']:.2f}s)")
            print(f"🎯 Best Coherence: {best_coherence['Model']} ({best_coherence['Coherence (C_v)']:.4f})")
        else:
            print("\n--- No models completed successfully. ---")

        print("="*80)


def main():
    """Main function to configure and run the comparison benchmark."""
    options = {
        "LEMMATIZE": True,
        "N_TOPICS": 15,
        "DESIRED_TOPIC_COUNT": 5,
        "tokenizer_type": "bpe",
        "nmf_type": "nmf",
        "LANGUAGE": "EN",
        "separator": ",",
        "tokenizer": None,
        "emoji_map": False,
        "save_to_db": False,
        "word_pairs_out": False,
        "topic_distribution": False,
    }

    filepath = "../../veri_setleri/bbc_news.csv"
    data_name = os.path.splitext(os.path.basename(filepath))[0]
    table_name = f"{data_name}_5-way_comparison_fixed"
    desired_columns = "text"

    benchmark = ComparisonBenchmark()
    result = benchmark.run_comparison_benchmark(filepath, table_name, desired_columns, options)

    if result["state"] == "SUCCESS":
        print(f"\n✅ Benchmark completed successfully!")
        print(f"   Output directory: {result['output_directory']}")
        print(f"   Total time: {result['total_time']:.2f}s")
    else:
        print(f"\n❌ Benchmark failed: {result['message']}")


if __name__ == "__main__":
    main()