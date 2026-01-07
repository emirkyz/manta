#!/usr/bin/env python3
"""
MANTA vs Gensim NMF vs Sklearn NMF Performance Comparison Script

This script benchmarks three NMF implementations with identical preprocessing:
- MANTA NMF (Custom implementation)
- Gensim NMF 
- Scikit-learn NMF

All models use the same preprocessing pipeline and are evaluated on:
- Speed (execution time)
- Coherence score (c_v metric)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
from scipy import sparse

# NMF implementations
from sklearn.decomposition import NMF as SklearnNMF
import gensim
import gensim.corpora as corpora
from gensim.models import Nmf as GensimNMF
from gensim.models.coherencemodel import CoherenceModel

# Add parent directory to path for MANTA imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# Import MANTA components
from manta._functions.common_language.emoji_processor import EmojiMap
from manta._functions.english.english_entry import process_english_file
from manta._functions.nmf import run_nmf
from manta._functions.turkish.turkish_entry import process_turkish_file
from manta._functions.turkish.turkish_tokenizer_factory import init_tokenizer
from manta.utils.database.database_manager import DatabaseManager
from manta.utils.console.console_manager import ConsoleManager


class NMFComparisonBenchmark:
    """
    Benchmark class for comparing MANTA, Gensim, and Sklearn NMF implementations.
    
    Ensures consistent preprocessing and evaluation across all three methods.
    """

    def __init__(self, console: Optional[ConsoleManager] = None):
        self.console = console or ConsoleManager()
        self.timing_results = {}
        self.preprocessing_data = {}
        self.results = {}

    def record_time(self, step_name: str, start_time: float) -> float:
        """Record timing for a processing step."""
        elapsed = time.time() - start_time
        self.timing_results[step_name] = elapsed
        self.console.print_status(f"{step_name} completed in {elapsed:.2f}s", "info")
        return elapsed

    def load_and_preprocess_data(self, filepath: str, desired_columns: str, options: Dict[str, Any]) -> pd.DataFrame:
        """Load and preprocess data using MANTA's approach for consistency."""
        step_start = time.time()
        self.console.print_status("Loading input file...", "processing")

        # Load data
        if str(filepath).endswith(".csv"):
            df = pd.read_csv(filepath, encoding="utf-8", sep=options.get("separator", ","), 
                           engine="python", on_bad_lines="skip")
        elif str(filepath).endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        self.record_time("Data Loading", step_start)

        # Preprocess data
        step_start = time.time()
        self.console.print_status("Preprocessing data...", "processing")
        
        if desired_columns not in df.columns:
            raise KeyError(f"Column '{desired_columns}' not found. Available: {', '.join(df.columns)}")

        df = df[[desired_columns]]
        initial_count = len(df)
        df = df.drop_duplicates().dropna()

        if len(df) == 0:
            raise ValueError("No data remaining after removing duplicates and null values")
        
        if len(df) < initial_count * 0.1:
            self.console.print_status(f"Warning: Only {len(df)} rows remain from {initial_count}", "warning")

        self.console.print_status(f"Preprocessed dataset has {len(df)} rows", "info")
        self.record_time("Data Preprocessing", step_start)
        return df

    def perform_text_processing(self, df: pd.DataFrame, desired_columns: str, options: Dict[str, Any]):
        """Perform language-specific text processing using MANTA's pipeline."""
        step_start = time.time()
        self.console.print_status(f"Starting text processing ({options['LANGUAGE']})...", "processing")

        if options["LANGUAGE"] == "TR":
            tdm_tfidf, vocab, counterized_data, text_array, tokenizer, _ = process_turkish_file(
                df, desired_columns, options["tokenizer"], 
                tokenizer_type=options["tokenizer_type"], emoji_map=None
            )
            options["tokenizer"] = tokenizer
        elif options["LANGUAGE"] == "EN":
            tdm_tfidf, vocab, counterized_data, text_array, _ = process_english_file(
                df, desired_columns, options["LEMMATIZE"], emoji_map=None
            )
        else:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}")

        self.console.print_status("Text processing completed", "success")
        self.record_time("Text Processing", step_start)

        # Prepare tokenized texts for coherence calculation
        texts_as_tokens = []
        if isinstance(text_array[0], str):
            texts_as_tokens = [doc.split() for doc in text_array]
        else:
            texts_as_tokens = text_array

        # Create Gensim dictionary for coherence
        gensim_dictionary = corpora.Dictionary(texts_as_tokens)
        gensim_dictionary.filter_extremes(no_below=1, no_above=0.95, keep_n=None)

        # Convert vocab to list format
        if isinstance(vocab, dict):
            vocab_list = [word for word, idx in sorted(vocab.items(), key=lambda x: x[1])]
        else:
            vocab_list = vocab

        self.preprocessing_data = {
            'tdm_tfidf': tdm_tfidf,
            'vocab': vocab_list,
            'text_array': text_array,
            'texts_as_tokens': texts_as_tokens,
            'gensim_dictionary': gensim_dictionary,
            'options': options.copy()
        }

    def extract_topics_from_matrix(self, topic_word_matrix: np.ndarray, vocab: List[str], num_words: int) -> Dict[str, Dict[str, float]]:
        """Extract topic words from topic-word matrix (e.g., sklearn's components_)."""
        topic_word_scores = {}
        for topic_idx, topic in enumerate(topic_word_matrix):
            topic_name = f"Topic_{topic_idx + 1}"
            top_word_indices = topic.argsort()[-num_words:][::-1]
            word_scores = {}
            for i in top_word_indices:
                if i < len(vocab):
                    word_scores[vocab[i]] = float(topic[i])
            topic_word_scores[topic_name] = word_scores
        return topic_word_scores

    def create_gensim_inputs(self):
        """Create Gensim dictionary and corpus from preprocessed data."""
        if 'gensim_corpus' not in self.preprocessing_data:
            self.console.print_status("Creating Gensim dictionary and corpus...", "processing")
            
            texts_as_tokens = self.preprocessing_data['texts_as_tokens']
            id2word = corpora.Dictionary(texts_as_tokens)
            id2word.filter_extremes(no_below=1, no_above=0.95, keep_n=None)
            
            corpus = [id2word.doc2bow(text) for text in texts_as_tokens]
            
            self.preprocessing_data['gensim_corpus_id2word'] = id2word
            self.preprocessing_data['gensim_corpus'] = corpus

        return self.preprocessing_data['gensim_corpus_id2word'], self.preprocessing_data['gensim_corpus']

    def calculate_coherence_score(self, topics: Dict[str, Dict[str, float]], model_name: str) -> float:
        """Calculate coherence score using Gensim's CoherenceModel."""
        try:
            self.console.print_status(f"Calculating coherence for {model_name}...", "processing")
            
            texts = self.preprocessing_data['texts_as_tokens']
            gensim_dictionary = self.preprocessing_data['gensim_dictionary']
            
            # Extract topic words (just the words, not the scores)
            topic_words = []
            for topic_dict in topics.values():
                words = list(topic_dict.keys())
                topic_words.append(words)

            # Calculate coherence using c_v metric
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

    def benchmark_manta_nmf(self) -> Dict[str, Any]:
        """Benchmark MANTA's NMF implementation."""
        step_start = time.time()
        self.console.print_status("Benchmarking MANTA NMF...", "processing")

        try:
            options = self.preprocessing_data['options']
            vocab = self.preprocessing_data['vocab']
            tdm_tfidf = self.preprocessing_data['tdm_tfidf']
            
            num_topics = int(options["DESIRED_TOPIC_COUNT"])
            num_words = int(options["N_TOPICS"])

            # Run MANTA NMF
            nmf_output = run_nmf(
                num_of_topics=num_topics,
                sparse_matrix=tdm_tfidf,
                nmf_method=options["nmf_type"]
            )

            # Extract topics
            topic_word_scores = self.extract_topics_from_matrix(nmf_output["H"], vocab, num_words)
            
            # Calculate coherence
            coherence_score = self.calculate_coherence_score(topic_word_scores, "MANTA NMF")
            elapsed_time = self.record_time("MANTA NMF", step_start)

            self.console.print_status(
                f"MANTA NMF: {elapsed_time:.2f}s, Coherence: {coherence_score:.4f}", "success"
            )

            return {
                "time": elapsed_time,
                "coherence": coherence_score,
                "topics": topic_word_scores,
                "model_output": nmf_output
            }

        except Exception as e:
            elapsed_time = self.record_time("MANTA NMF (Failed)", step_start)
            self.console.print_status(f"MANTA NMF failed: {e}", "error")
            return {
                "time": elapsed_time,
                "coherence": 0.0,
                "topics": {},
                "error": str(e)
            }

    def benchmark_sklearn_nmf(self) -> Dict[str, Any]:
        """Benchmark Scikit-learn's NMF implementation."""
        step_start = time.time()
        self.console.print_status("Benchmarking Scikit-learn NMF...", "processing")

        try:
            options = self.preprocessing_data['options']
            vocab = self.preprocessing_data['vocab']
            tdm_tfidf = self.preprocessing_data['tdm_tfidf']
            
            num_topics = int(options["DESIRED_TOPIC_COUNT"])
            num_words = int(options["N_TOPICS"])

            # Convert sparse matrix to dense array
            if sparse.issparse(tdm_tfidf):
                tdm_array = tdm_tfidf.toarray()
            else:
                tdm_array = tdm_tfidf

            # Ensure non-negative values for NMF
            tdm_array = np.abs(tdm_array)

            # Run Sklearn NMF
            model = SklearnNMF(
                n_components=num_topics,
                random_state=42,
                init='nndsvd',
                max_iter=1000,
                solver='cd',
                alpha_W=0.01,
                alpha_H=0.01,
                l1_ratio=0.0
            )
            model.fit(tdm_array)

            # Extract topics
            topic_word_scores = self.extract_topics_from_matrix(model.components_, vocab, num_words)
            
            # Calculate coherence
            coherence_score = self.calculate_coherence_score(topic_word_scores, "Sklearn NMF")
            elapsed_time = self.record_time("Sklearn NMF", step_start)

            self.console.print_status(
                f"Sklearn NMF: {elapsed_time:.2f}s, Coherence: {coherence_score:.4f}", "success"
            )

            return {
                "time": elapsed_time,
                "coherence": coherence_score,
                "topics": topic_word_scores,
                "model": model
            }

        except Exception as e:
            elapsed_time = self.record_time("Sklearn NMF (Failed)", step_start)
            self.console.print_status(f"Sklearn NMF failed: {e}", "error")
            return {
                "time": elapsed_time,
                "coherence": 0.0,
                "topics": {},
                "error": str(e)
            }

    def benchmark_gensim_nmf(self) -> Dict[str, Any]:
        """Benchmark Gensim's NMF implementation."""
        step_start = time.time()
        self.console.print_status("Benchmarking Gensim NMF...", "processing")

        try:
            options = self.preprocessing_data['options']
            num_topics = int(options["DESIRED_TOPIC_COUNT"])
            num_words = int(options["N_TOPICS"])

            # Create Gensim inputs
            id2word, corpus = self.create_gensim_inputs()

            # Apply TF-IDF transformation for NMF
            tfidf = gensim.models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]

            # Run Gensim NMF
            model = GensimNMF(
                corpus=corpus_tfidf,
                num_topics=num_topics,
                id2word=id2word,
                random_state=42,
                passes=50,
                minimum_probability=0.001,
                normalize=True,
                chunksize=100
            )

            # Extract topics
            topics = model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
            topic_word_scores = {}
            for topic_id, topic_words in topics:
                topic_name = f"Topic_{topic_id + 1}"
                word_scores = {}
                for word, score in topic_words:
                    word_scores[word] = float(score)
                topic_word_scores[topic_name] = word_scores

            # Calculate coherence
            coherence_score = self.calculate_coherence_score(topic_word_scores, "Gensim NMF")
            elapsed_time = self.record_time("Gensim NMF", step_start)

            self.console.print_status(
                f"Gensim NMF: {elapsed_time:.2f}s, Coherence: {coherence_score:.4f}", "success"
            )

            return {
                "time": elapsed_time,
                "coherence": coherence_score,
                "topics": topic_word_scores,
                "model": model
            }

        except Exception as e:
            elapsed_time = self.record_time("Gensim NMF (Failed)", step_start)
            self.console.print_status(f"Gensim NMF failed: {e}", "error")
            return {
                "time": elapsed_time,
                "coherence": 0.0,
                "topics": {},
                "error": str(e)
            }

    def run_comparison(self, filepath: str, table_name: str, desired_columns: str, options: Dict[str, Any], output_base_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run complete NMF comparison benchmark."""
        try:
            self.console.start_timing()
            self.console.print_header("NMF Methods Comparison Benchmark", "MANTA vs Gensim vs Scikit-learn")
            self.console.display_config(options, filepath, desired_columns, table_name)

            # Setup
            setup_start = time.time()
            db_config = DatabaseManager.initialize_database_config(output_base_dir)
            table_output_dir = db_config.output_dir / table_name
            table_output_dir.mkdir(parents=True, exist_ok=True)
            self.record_time("Setup", setup_start)

            # Initialize tokenizer if needed
            init_start = time.time()
            if not options.get("tokenizer"):
                options["tokenizer"] = init_tokenizer(tokenizer_type=options["tokenizer_type"])
            options["emoji_map"] = EmojiMap() if options.get("emoji_map") else None
            self.record_time("Initialization", init_start)

            # Load and preprocess data
            df = self.load_and_preprocess_data(filepath, desired_columns, options)
            self.perform_text_processing(df, desired_columns, options)

            # Benchmark all three NMF methods
            self.results = {
                "MANTA NMF": self.benchmark_manta_nmf(),
                "Sklearn NMF": self.benchmark_sklearn_nmf(),
                "Gensim NMF": self.benchmark_gensim_nmf()
            }

            # Print comparison summary
            self.print_comparison_summary()

            # Save results
            results_path = table_output_dir / "nmf_comparison_results.json"
            
            def serialize_numpy(obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            # Create serializable results
            serializable_results = {}
            for model_name, result in self.results.items():
                serializable_result = {
                    "time": result["time"],
                    "coherence": result["coherence"],
                    "topics": result["topics"]
                }
                if "error" in result:
                    serializable_result["error"] = result["error"]
                serializable_results[model_name] = serializable_result

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({
                    "benchmark_results": serializable_results,
                    "preprocessing_info": {
                        "dataset_size": len(df),
                        "vocabulary_size": len(self.preprocessing_data['vocab']),
                        "language": options["LANGUAGE"],
                        "num_topics": options["DESIRED_TOPIC_COUNT"],
                        "words_per_topic": options["N_TOPICS"]
                    }
                }, f, ensure_ascii=False, indent=2, default=serialize_numpy)

            total_time = self.console.get_total_time()
            self.console.print_status(f"Results saved to {results_path}", "info")
            self.console.print_status(f"Total benchmark time: {total_time:.2f}s", "success")

            return {
                "state": "SUCCESS",
                "results": serializable_results,
                "total_time": total_time,
                "output_directory": str(table_output_dir)
            }

        except Exception as e:
            import traceback
            self.console.print_status(f"Benchmark failed: {e}", "error")
            print(f"Full error trace: {traceback.format_exc()}")
            return {"state": "FAILURE", "message": str(e)}

    def print_comparison_summary(self):
        """Print detailed comparison summary."""
        print("\n" + "="*80)
        print("NMF METHODS COMPARISON SUMMARY")
        print("="*80)

        # Prepare summary data
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                "Model": name,
                "Time (s)": result.get("time", float('inf')),
                "Coherence (C_v)": result.get("coherence", 0.0),
                "Status": "Failed" if "error" in result else "Success"
            })

        df = pd.DataFrame(summary_data)

        # Sort and display results
        print("\n--- Results sorted by Speed (fastest first) ---")
        print(df.sort_values(by="Time (s)").to_string(index=False, float_format="%.4f"))

        print("\n--- Results sorted by Coherence (best first) ---")
        print(df.sort_values(by="Coherence (C_v)", ascending=False).to_string(index=False, float_format="%.4f"))

        # Determine winners
        successful_runs = df[df['Status'] == 'Success']
        if not successful_runs.empty:
            fastest = successful_runs.loc[successful_runs["Time (s)"].idxmin()]
            best_coherence = successful_runs.loc[successful_runs["Coherence (C_v)"].idxmax()]

            print("\n--- WINNERS ---")
            print(f"🚀 Fastest Model : {fastest['Model']} ({fastest['Time (s)']:.2f}s)")
            print(f"🎯 Best Coherence: {best_coherence['Model']} ({best_coherence['Coherence (C_v)']:.4f})")
            
            # Performance ratio analysis
            if len(successful_runs) > 1:
                print("\n--- Performance Analysis ---")
                time_ratio = successful_runs["Time (s)"].max() / successful_runs["Time (s)"].min()
                coherence_diff = successful_runs["Coherence (C_v)"].max() - successful_runs["Coherence (C_v)"].min()
                print(f"Speed difference: {time_ratio:.2f}x between fastest and slowest")
                print(f"Coherence difference: {coherence_diff:.4f} between best and worst")
        else:
            print("\n--- No models completed successfully ---")

        # Show sample topics from best performing model
        if successful_runs.empty:
            return
            
        best_model = best_coherence['Model']
        print(f"\n--- Sample Topics from {best_model} ---")
        topics = self.results[best_model]["topics"]
        
        for i, (topic_name, words) in enumerate(list(topics.items())[:3]):  # Show first 3 topics
            top_words = list(words.keys())[:5]  # Show top 5 words
            print(f"{topic_name}: {', '.join(top_words)}")

        print("="*80)


def main():
    """Main function to run the NMF comparison benchmark."""
    # Configuration
    options = {
        "LEMMATIZE": True,
        "N_TOPICS": 15,  # Words per topic
        "DESIRED_TOPIC_COUNT": 5,  # Number of topics
        "tokenizer_type": "bpe",
        "nmf_type": "nmf",
        "LANGUAGE": "EN",  # Change to "TR" for Turkish
        "separator": ",",
        "tokenizer": None,
        "emoji_map": False,
        "save_to_db": False
    }

    # File configuration
    filepath = "../../datasets/bbc_news.csv"  # Update path as needed
    data_name = os.path.splitext(os.path.basename(filepath))[0]
    table_name = f"{data_name}_nmf_comparison_{options['DESIRED_TOPIC_COUNT']}_topics"
    desired_columns = "text"  # Update column name as needed

    # Run benchmark
    benchmark = NMFComparisonBenchmark()
    result = benchmark.run_comparison(filepath, table_name, desired_columns, options)

    if result["state"] == "SUCCESS":
        print(f"\n✅ NMF Comparison benchmark completed successfully!")
        print(f"   Output directory: {result['output_directory']}")
        print(f"   Total time: {result['total_time']:.2f}s")
    else:
        print(f"\n❌ Benchmark failed: {result['message']}")


if __name__ == "__main__":
    main()