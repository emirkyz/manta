import time

from .english_preprocessor import clean_english_text
from .english_text_encoder import counterize_english
from .english_vocabulary import create_english_vocab
from ..._functions.tfidf import tf_idf_english
from ..common_language.ngram_bpe import WordPairBPE

START_TIME = time.time()


def process_english_file(df, desired_columns: str, lemmatize: bool, emoji_map=None,
                        enable_ngram_bpe=False, ngram_vocab_limit=10000, min_pair_frequency=2):
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
        enable_ngram_bpe (bool): Whether to apply n-gram BPE after counterization
        ngram_vocab_limit (int): Maximum vocabulary size for n-gram BPE
        min_pair_frequency (int): Minimum frequency threshold for pair merging

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
    print(f"Preprocess completed in {time.time() - START_TIME:.2f} seconds")
    vocab, N = create_english_vocab(text_array, desired_columns, lemmatize=lemmatize)
    counterized_data = counterize_english(vocab=vocab, data=text_array,lemmatize=lemmatize)

    # Apply n-gram BPE if enabled
    if True : # enable_ngram_bpe
        print(f"Applying n-gram BPE with vocab limit: {ngram_vocab_limit}")
        bpe_start_time = time.time()
        target_vocab_size = len(vocab) + min(200, len(vocab) // 5)
        target_vocab_size = len(vocab) + 300
        # Initialize and train BPE encoder
        ngram_bpe = WordPairBPE(vocab_limit=target_vocab_size, min_pair_frequency=min_pair_frequency)
        counterized_data = ngram_bpe.fit_optimized(counterized_data, len(vocab), vocab)

        # Update vocabulary with n-gram information
        ngram_info = ngram_bpe.get_ngram_vocab_info()
        print(f"N-gram BPE completed in {time.time() - bpe_start_time:.2f} seconds")
        print(f"Created {ngram_info['ngrams_created']} n-gram combinations")
        print(f"Vocabulary expanded from {ngram_info['original_vocab_size']} to {ngram_info['final_vocab_size']}")

        # Extend vocabulary with n-gram entries (for compatibility)
        extended_vocab = vocab[:]  # Copy original vocab
        for i in range(len(vocab), ngram_bpe.current_vocab_size):
            if i in ngram_bpe.id_to_pair:
                ngram_meaning = ngram_bpe.reconstruct_ngram_meaning(i, vocab)
                extended_vocab.append(ngram_meaning)
            else:
                extended_vocab.append(f"NGRAM_{i}")

        vocab = extended_vocab

        # Save n-grams to JSON file
        #try:
        #    output_dir = "Output"
        #    ngram_file = ngram_bpe.save_ngrams_to_json("english_ngrams.json", vocab, output_dir)
        #    print(f"English n-grams analysis saved to: {ngram_file}")
        #except Exception as e:
        #    print(f"Warning: Could not save n-grams file: {e}")

    # tfidf
    tdm = tf_idf_english(N, vocab=vocab, data=counterized_data, fieldname=desired_columns, output_dir=None,
                         lemmatize=lemmatize)

    return tdm, vocab, counterized_data, text_array, emoji_map
