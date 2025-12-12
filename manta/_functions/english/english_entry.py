import time

from .english_preprocessor import clean_english_text
from .english_text_encoder import counterize_english
from .english_vocabulary import create_english_vocab
from ..._functions.tfidf import tf_idf_english
from ..common_language.ngram_bpe import WordPairBPE
from ..common_language.ngram_wordpiece import WordPieceNGram

START_TIME = time.time()


def process_english_file(df, desired_columns: str, lemmatize: bool, emoji_map=None,
                        n_gram_discover_count=None, ngram_vocab_limit=10000, min_pair_frequency=2,
                        ngram_algorithm="wordpiece", min_likelihood_score=0.0, pagerank_weights=None):
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
        enable_ngram_bpe (bool): Whether to apply n-gram algorithm after counterization
        ngram_vocab_limit (int): Maximum vocabulary size for n-gram algorithm
        min_pair_frequency (int): Minimum frequency threshold for pair merging (BPE only)
        ngram_algorithm (str): Choice of n-gram algorithm: "bpe" or "wordpiece" (default: "wordpiece")
        min_likelihood_score (float): Minimum likelihood threshold for pair merging (WordPiece only, default: 0.0)
        pagerank_weights (numpy.ndarray, optional): Per-document weights for TF-IDF boosting.
            Array of shape (N,) with weights typically in range [1, 2].

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
    text_array = clean_english_text(metin=df[desired_columns].values, lemmatize=lemmatize, emoji_map=emoji_map)

    # Track text processing results
    original_doc_count = len(df[desired_columns].values)
    non_empty_docs = sum(1 for text in text_array if text and text.strip())
    empty_docs = original_doc_count - non_empty_docs
    print(f"\n[TEXT PROCESSING] Document statistics after cleaning:")
    print(f"  Original documents: {original_doc_count}")
    print(f"  Non-empty after cleaning: {non_empty_docs}")
    print(f"  Empty after cleaning: {empty_docs}")
    if empty_docs > 0:
        percent_empty = (empty_docs / original_doc_count * 100) if original_doc_count > 0 else 0
        print(f"  WARNING: {empty_docs} documents ({percent_empty:.1f}%) became empty during text cleaning!")

    print(f"Preprocess completed in {time.time() - START_TIME:.2f} seconds")
    vocab, N = create_english_vocab(text_array, desired_columns, lemmatize=lemmatize)
    counterized_data = counterize_english(vocab=vocab, data=text_array,lemmatize=lemmatize)

    # Apply n-gram algorithm if enabled
    if n_gram_discover_count is not None : # enable_ngram_bpe
        #target_vocab_size = len(vocab) + min(200, len(vocab) // 5)
        target_vocab_size = len(vocab) + n_gram_discover_count
        ngram_algorithm = "bpe"
        if ngram_algorithm.lower() == "wordpiece":
            print(f"Applying n-gram WordPiece with vocab limit: {target_vocab_size}")
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
            print(f"N-gram WordPiece completed in {time.time() - ngram_start_time:.2f} seconds")
            print(f"Created {ngram_info['ngrams_created']} n-gram combinations")
            print(f"Vocabulary expanded from {ngram_info['original_vocab_size']} to {ngram_info['final_vocab_size']}")

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
            print(f"Applying n-gram BPE with vocab limit: {target_vocab_size}")
            ngram_start_time = time.time()

            # Initialize and train BPE encoder
            ngram_encoder = WordPairBPE(
                vocab_limit=target_vocab_size,
                min_pair_frequency=min_pair_frequency,
                verbose=False
            )
            counterized_data = ngram_encoder.fit_optimized(counterized_data, len(vocab), vocab)

            # Update vocabulary with n-gram information
            ngram_info = ngram_encoder.get_ngram_vocab_info()
            print(f"N-gram BPE completed in {time.time() - ngram_start_time:.2f} seconds")
            print(f"Created {ngram_info['ngrams_created']} n-gram combinations")
            print(f"Vocabulary expanded from {ngram_info['original_vocab_size']} to {ngram_info['final_vocab_size']}")

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

        # Reconstruct text_array to reflect n-gram merges for coherence calculation
        text_array = [
            " ".join([vocab[token_id] if token_id < len(vocab) else f"UNK_{token_id}"
                      for token_id in doc])
            for doc in counterized_data
        ]
        print(f"Text array reconstructed with n-gram tokens for coherence calculation")

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

    print(f"TF-IDF shape = {tdm.shape}, the amount of words = {tdm.shape[1]}, and the amount of documents = {tdm.shape[0]} ")
    return tdm, vocab, counterized_data, text_array, emoji_map
