import time
from math import sqrt
from typing import List, Tuple, Dict, Optional

import numpy as np

from .english_preprocessor import clean_english_text
from .english_text_encoder import counterize_english
from .english_vocabulary import create_english_vocab
from ..._functions.tfidf import tf_idf_english
from ..common_language.ngram_bpe import WordPairBPE
from ..common_language.ngram_wordpiece import WordPieceNGram
from ...utils.console.console_manager import ConsoleManager, get_console

START_TIME = time.time()


def is_pure_number(token: str) -> bool:
    """
    Check if a token is a pure number (standalone number that didn't merge with letters).

    Args:
        token: The token string to check

    Returns:
        True if token is purely numeric (e.g., "123", "2020"),
        False if it contains letters or merged patterns (e.g., "covid_19", "120_mg")
    """
    if '_' in token:
        # This is a merged BPE token - check if any part contains letters
        parts = token.split('_')
        # If ANY part contains a letter, keep this token
        for part in parts:
            if any(c.isalpha() for c in part):
                return False  # Mixed alphanumeric - keep it
        return True  # All parts are numeric - filter it out
    else:
        # Simple token - check if purely numeric
        return token.isdigit()


def filter_pure_numbers_from_vocab(vocab: List[str], counterized_data: List) -> Tuple[List[str], List, Dict[int, int], int]:
    """
    Filter out pure standalone numbers from vocabulary and counterized data after BPE.

    This preserves merged tokens like "covid_19", "type_2", "120_mg" while removing
    standalone numbers like "123", "2020" that didn't merge with any word.

    Args:
        vocab: Extended vocabulary list (including BPE-created n-grams)
        counterized_data: List of documents as lists/arrays of token IDs

    Returns:
        tuple: (filtered_vocab, filtered_counterized_data, id_mapping, num_filtered)
            - filtered_vocab: Vocabulary with pure numbers removed
            - filtered_counterized_data: Data with number token IDs remapped
            - id_mapping: Dict mapping old IDs to new IDs (for debugging)
            - num_filtered: Count of filtered tokens
    """
    # Identify tokens to keep (not pure numbers)
    tokens_to_keep = []
    old_to_new_id = {}
    new_id = 0

    for old_id, token in enumerate(vocab):
        if not is_pure_number(token):
            tokens_to_keep.append(token)
            old_to_new_id[old_id] = new_id
            new_id += 1
        # else: token is a pure number, skip it (no mapping)

    filtered_vocab = tokens_to_keep

    # Remap counterized data, removing tokens that are pure numbers
    filtered_counterized_data = []
    for doc in counterized_data:
        if isinstance(doc, np.ndarray):
            # Filter out IDs not in mapping and remap remaining IDs
            new_doc = [old_to_new_id[int(token_id)] for token_id in doc if int(token_id) in old_to_new_id]
            filtered_counterized_data.append(np.array(new_doc, dtype=doc.dtype))
        else:
            new_doc = [old_to_new_id[token_id] for token_id in doc if token_id in old_to_new_id]
            filtered_counterized_data.append(new_doc)

    # Count filtered tokens
    num_filtered = len(vocab) - len(filtered_vocab)

    return filtered_vocab, filtered_counterized_data, old_to_new_id, num_filtered


def calculate_auto_ngram_count(vocab_size: int, k: float = 0.5) -> int:
    """
    Calculate the number of n-grams to discover based on vocabulary size.

    Formula: n_grams = int(sqrt(vocab_size) * k)

    Args:
        vocab_size: Current vocabulary size before BPE
        k: Scaling constant (default: 0.5)
           - k=0.5: Conservative (smaller vocab expansion)
           - k=1.0: Moderate (sqrt of vocab_size)
           - k=2.0: Aggressive (larger vocab expansion)

    Returns:
        int: Number of n-grams to discover

    Examples:
        vocab_size=10000, k=0.5 -> 50 n-grams
        vocab_size=10000, k=1.0 -> 100 n-grams
        vocab_size=50000, k=0.5 -> 112 n-grams
        vocab_size=50000, k=1.0 -> 224 n-grams
    """
    ngram_count = int(sqrt(vocab_size) * k)
    # Ensure at least some n-grams are discovered
    return max(ngram_count, 10)


def process_english_file(df, desired_columns: str, lemmatize: bool, emoji_map=None,
                        n_gram_discover_count=None, ngram_vocab_limit=10000, min_pair_frequency=2,
                        ngram_algorithm="wordpiece", min_likelihood_score=0.0, pagerank_weights=None,
                        keep_numbers=False, ngram_auto_k=0.5, filter_standalone_numbers=True,
                        use_pmi=True, console: Optional[ConsoleManager] = None):
    """
    Process English text data for topic modeling using NMF.

    This function performs text preprocessing and TF-IDF transformation specifically
    for English language texts. It creates a vocabulary dictionary and transforms
    the text data into numerical format suitable for topic modeling.

    Args:
        df (pd.DataFrame): Input DataFrame containing English text data
        desired_columns (str): Name of the column containing text to analyze
        lemmatize (bool): Whether to apply lemmatization during preprocessing.
                         If True, words are reduced to their base forms
        emoji_map: Emoji mapping instance for preprocessing
        n_gram_discover_count: Number of n-grams to discover, "auto" for automatic, or None to disable
        ngram_vocab_limit (int): Maximum vocabulary size for n-gram algorithm
        min_pair_frequency (int): Minimum frequency threshold for pair merging (BPE only)
        ngram_algorithm (str): Choice of n-gram algorithm: "bpe" or "wordpiece" (default: "wordpiece")
        min_likelihood_score (float): Minimum likelihood threshold for pair merging (WordPiece only, default: 0.0)
        pagerank_weights (numpy.ndarray, optional): Per-document weights for TF-IDF boosting.
            Array of shape (N,) with weights typically in range [1, 2].
        keep_numbers (bool): If True, preserve numbers during preprocessing for BPE merging.
                            Numbers will participate in BPE then unmerged ones are filtered.
        ngram_auto_k (float): Scaling constant for auto n-gram count formula (default: 0.5).
                             Only used when n_gram_discover_count is "auto".
        filter_standalone_numbers (bool): If True and keep_numbers is True, filter out
                                         standalone numbers after BPE completes.
        use_pmi (bool): If True and keep_numbers is True, use PMI (Pointwise Mutual
                       Information) scoring for BPE instead of raw frequency. This helps
                       number-word pairs compete fairly with more frequent word-word pairs.
        console (ConsoleManager, optional): Console manager for output. If None, uses global console.

    Returns:
        tuple: A tuple containing:
            - tdm (scipy.sparse matrix): Term-document matrix (TF-IDF transformed)
            - vocab (dict): Vocabulary dictionary mapping words to indices (updated with n-grams if enabled)
            - counterized_data (list): Counterized numerical data (with n-grams if enabled)
            - text_array (list): Preprocessed text array
            - emoji_map: Updated emoji mapping

    Raises:
        KeyError: If desired_columns is not found in the DataFrame
        ValueError: If the DataFrame is empty or contains no valid text data
    """
    # Initialize console
    _console = console or get_console()

    # Determine if we need to keep numbers for BPE merging
    should_keep_numbers = keep_numbers and (n_gram_discover_count is not None)

    text_array = clean_english_text(metin=df[desired_columns].values, lemmatize=lemmatize,
                                    emoji_map=emoji_map, keep_numbers=should_keep_numbers)

    if should_keep_numbers:
        _console.print_debug("Keeping numbers for BPE merging (will filter unmerged numbers after BPE)", tag="NUMBER PRESERVATION")

    # Track text processing results
    original_doc_count = len(df[desired_columns].values)
    non_empty_docs = sum(1 for text in text_array if text and text.strip())
    empty_docs = original_doc_count - non_empty_docs
    _console.print_debug("Document statistics after cleaning:", tag="TEXT PROCESSING")
    _console.print_debug(f"  Original documents: {original_doc_count}", tag="TEXT PROCESSING")
    _console.print_debug(f"  Non-empty after cleaning: {non_empty_docs}", tag="TEXT PROCESSING")
    _console.print_debug(f"  Empty after cleaning: {empty_docs}", tag="TEXT PROCESSING")
    if empty_docs > 0:
        percent_empty = (empty_docs / original_doc_count * 100) if original_doc_count > 0 else 0
        _console.print_warning(f"{empty_docs} documents ({percent_empty:.1f}%) became empty during text cleaning!", tag="TEXT PROCESSING")

    _console.print_debug(f"Preprocess completed in {time.time() - START_TIME:.2f} seconds", tag="TEXT PROCESSING")
    vocab, N = create_english_vocab(text_array, desired_columns, lemmatize=lemmatize)
    counterized_data = counterize_english(vocab=vocab, data=text_array,lemmatize=lemmatize)

    # Apply n-gram algorithm if enabled
    if n_gram_discover_count is not None:  # enable_ngram_bpe
        # Calculate auto n-gram count if requested
        if n_gram_discover_count == "auto":
            n_gram_discover_count = calculate_auto_ngram_count(len(vocab), k=ngram_auto_k)
            _console.print_debug(f"Calculated {n_gram_discover_count} n-grams to discover "
                  f"(vocab_size={len(vocab)}, k={ngram_auto_k})", tag="AUTO N-GRAM")

        target_vocab_size = len(vocab) + n_gram_discover_count
        ngram_algorithm = "bpe"
        if ngram_algorithm.lower() == "wordpiece":
            _console.print_debug(f"Applying n-gram WordPiece with vocab limit: {target_vocab_size}", tag="N-GRAM")
            ngram_start_time = time.time()

            # Initialize and train WordPiece encoder
            ngram_encoder = WordPieceNGram(
                vocab_limit=target_vocab_size,
                min_likelihood_score=min_likelihood_score,
                smoothing=1e-10,
                verbose=False
            )
            counterized_data = ngram_encoder.fit_optimized(counterized_data, len(vocab), vocab)

            # Update vocabulary with n-gram information
            ngram_info = ngram_encoder.get_ngram_vocab_info()
            _console.print_debug(f"N-gram WordPiece completed in {time.time() - ngram_start_time:.2f} seconds", tag="N-GRAM")
            _console.print_debug(f"Created {ngram_info['ngrams_created']} n-gram combinations", tag="N-GRAM")
            _console.print_debug(f"Vocabulary expanded from {ngram_info['original_vocab_size']} to {ngram_info['final_vocab_size']}", tag="N-GRAM")

            # Extend vocabulary with n-gram entries (for compatibility)
            extended_vocab = vocab[:]  # Copy original vocab
            for i in range(len(vocab), ngram_encoder.current_vocab_size):
                if i in ngram_encoder.id_to_pair:
                    ngram_meaning = ngram_encoder.reconstruct_ngram_meaning(i, vocab)
                    extended_vocab.append(ngram_meaning)
                else:
                    extended_vocab.append(f"NGRAM_{i}")

            vocab = extended_vocab

        elif ngram_algorithm.lower() == "bpe":
            _console.print_debug(f"Applying n-gram BPE with vocab limit: {target_vocab_size}", tag="N-GRAM BPE")
            ngram_start_time = time.time()

            # Initialize and train BPE encoder
            # Enable PMI scoring when keep_numbers is True and use_pmi is True to give
            # number-word pairs fair treatment against more frequent word-word pairs
            ngram_encoder = WordPairBPE(
                vocab_limit=target_vocab_size,
                min_pair_frequency=min_pair_frequency,
                verbose=False,
                use_pmi=(should_keep_numbers and use_pmi),  # PMI helps number-word pairs compete fairly
                console=_console
            )
            counterized_data = ngram_encoder.fit_optimized(counterized_data, len(vocab), vocab)

            # Update vocabulary with n-gram information
            ngram_info = ngram_encoder.get_ngram_vocab_info()
            _console.print_debug(f"N-gram BPE completed in {time.time() - ngram_start_time:.2f} seconds", tag="N-GRAM BPE")
            _console.print_debug(f"Created {ngram_info['ngrams_created']} n-gram combinations", tag="N-GRAM BPE")
            _console.print_debug(f"Vocabulary expanded from {ngram_info['original_vocab_size']} to {ngram_info['final_vocab_size']}", tag="N-GRAM BPE")

            # Extend vocabulary with n-gram entries (for compatibility)
            extended_vocab = vocab[:]  # Copy original vocab
            for i in range(len(vocab), ngram_encoder.current_vocab_size):
                if i in ngram_encoder.id_to_pair:
                    ngram_meaning = ngram_encoder.reconstruct_ngram_meaning(i, vocab)
                    extended_vocab.append(ngram_meaning)
                else:
                    extended_vocab.append(f"NGRAM_{i}")

            vocab = extended_vocab
        else:
            raise ValueError(f"Unknown n-gram algorithm: {ngram_algorithm}. Must be 'bpe' or 'wordpiece'.")

        # Filter standalone numbers after BPE if enabled
        if should_keep_numbers and filter_standalone_numbers:
            pre_filter_vocab_size = len(vocab)
            vocab, counterized_data, _, num_filtered = filter_pure_numbers_from_vocab(
                vocab, counterized_data
            )
            _console.print_debug(f"Removed {num_filtered} standalone numbers from vocabulary", tag="NUMBER FILTER")
            _console.print_debug(f"Vocabulary size: {pre_filter_vocab_size} -> {len(vocab)}", tag="NUMBER FILTER")

        # Reconstruct text_array to reflect n-gram merges for coherence calculation
        text_array = [
            " ".join([vocab[token_id] if token_id < len(vocab) else f"UNK_{token_id}"
                      for token_id in doc])
            for doc in counterized_data
        ]
        _console.print_debug("Text array reconstructed with n-gram tokens for coherence calculation", tag="N-GRAM")

        # Save n-grams to JSON file (optional)
        #try:
        #    output_dir = "Output"
        #    algorithm_name = ngram_algorithm.lower()
        #    ngram_file = ngram_encoder.save_ngrams_to_json(f"english_ngrams_{algorithm_name}.json", vocab, output_dir)
        #    print(f"English n-grams analysis saved to: {ngram_file}")
        #except Exception as e:
        #    print(f"Warning: Could not save n-grams file: {e}")

    # tfidf
    tdm = tf_idf_english(N, vocab=vocab, data=counterized_data, fieldname=desired_columns, output_dir=None,
                         lemmatize=lemmatize, pagerank_weights=pagerank_weights)

    _console.print_debug(f"TF-IDF shape = {tdm.shape}, words = {tdm.shape[1]}, documents = {tdm.shape[0]}", tag="TF-IDF")
    return tdm, vocab, counterized_data, text_array, emoji_map
