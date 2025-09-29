from .turkish_preprocessor import clean_text_turkish
from .turkish_tokenizer_factory import init_tokenizer, train_tokenizer
from .turkish_text_encoder import counterize_turkish
from ..tfidf import tf_idf_turkish
from ..common_language.ngram_bpe import WordPairBPE
from ..common_language.ngram_tokenizer_wrapper import NgramTokenizerWrapper


def process_turkish_file(df, desired_columns: str, tokenizer=None, tokenizer_type=None, emoji_map=None,
                        enable_ngram_bpe=False, ngram_vocab_limit=10000, min_pair_frequency=2):
    """
    Process Turkish text data for topic modeling using NMF.

    This function performs text preprocessing, tokenization, and TF-IDF transformation
    specifically for Turkish language texts. It handles text cleaning, emoji mapping,
    tokenizer training, and vectorization.

    Args:
        df (pd.DataFrame): Input DataFrame containing Turkish text data
        desired_columns (str): Name of the column containing text to analyze
        tokenizer (optional): Pre-trained tokenizer instance. If None, a new tokenizer
                             will be initialized based on tokenizer_type
        tokenizer_type (str, optional): Type of tokenizer to use. Options: "bpe" or "wordpiece"
        emoji_map (EmojiMap, optional): Emoji mapping instance for emoji processing
        enable_ngram_bpe (bool): Whether to apply n-gram BPE on top of existing tokenization
        ngram_vocab_limit (int): Maximum vocabulary size for n-gram BPE
        min_pair_frequency (int): Minimum frequency threshold for pair merging

    Returns:
        tuple: A tuple containing:
            - tdm (scipy.sparse matrix): Term-document matrix (TF-IDF transformed)
            - vocabulary (list): Vocabulary list from the tokenizer (extended with n-grams if enabled)
            - counterized_data (list): Numerical representation of documents (with n-grams if enabled)
            - text_array (list): Cleaned text array
            - tokenizer: Trained tokenizer instance
            - emoji_map (EmojiMap): Emoji mapping instance used

    Raises:
        ValueError: If tokenizer_type is not supported
        KeyError: If desired_columns is not found in the DataFrame
    """

    text_array = clean_text_turkish(df, desired_columns, emoji_map=emoji_map)
    print(f"Number of documents: {len(text_array)}")

    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = init_tokenizer(tokenizer_type=tokenizer_type)

    # Train the tokenizer
    tokenizer = train_tokenizer(tokenizer, text_array, tokenizer_type=tokenizer_type)
    vocabulary = list(tokenizer.get_vocab().keys())

    # sayısallaştır (counterize)
    counterized_data = counterize_turkish(text_array, tokenizer)

    # Apply n-gram BPE on token IDs if enabled
    if enable_ngram_bpe:
        print(f"Applying token-level n-gram BPE with vocab limit: {ngram_vocab_limit}")
        import time
        bpe_start_time = time.time()

        # Initialize and train BPE encoder on token IDs
        ngram_bpe = WordPairBPE(vocab_limit=ngram_vocab_limit, min_pair_frequency=min_pair_frequency)
        counterized_data = ngram_bpe.fit(counterized_data, len(vocabulary), vocabulary)

        # Get n-gram information
        ngram_info = ngram_bpe.get_ngram_vocab_info()
        print(f"Token-level n-gram BPE completed in {time.time() - bpe_start_time:.2f} seconds")
        print(f"Created {ngram_info['ngrams_created']} token-pair n-grams")
        print(f"Vocabulary expanded from {ngram_info['original_vocab_size']} to {ngram_info['final_vocab_size']}")

        # Extend vocabulary with token-pair n-gram entries
        extended_vocabulary = vocabulary[:]  # Copy original vocabulary
        for i in range(len(vocabulary), ngram_bpe.current_vocab_size):
            if i in ngram_bpe.id_to_pair:
                # Reconstruct token pair meaning
                token1_id, token2_id = ngram_bpe.id_to_pair[i]
                if token1_id < len(vocabulary) and token2_id < len(vocabulary):
                    token_pair_meaning = f"{vocabulary[token1_id]}+{vocabulary[token2_id]}"
                else:
                    token_pair_meaning = ngram_bpe.reconstruct_ngram_meaning(i, vocabulary)
                extended_vocabulary.append(token_pair_meaning)
            else:
                extended_vocabulary.append(f"TOKEN_NGRAM_{i}")

        vocabulary = extended_vocabulary
        print(f"Final vocabulary size: {len(vocabulary)}")

        # Save n-grams to JSON file
        try:
            output_dir = "Output"
            ngram_file = ngram_bpe.save_ngrams_to_json("turkish_ngrams.json", vocabulary, output_dir)
            print(f"Turkish n-grams analysis saved to: {ngram_file}")
        except Exception as e:
            print(f"Warning: Could not save n-grams file: {e}")

        # Create wrapped tokenizer that supports n-gram tokens
        wrapped_tokenizer = NgramTokenizerWrapper(
            original_tokenizer=tokenizer,
            ngram_bpe=ngram_bpe,
            extended_vocabulary=extended_vocabulary
        )

        print("Created n-gram aware tokenizer wrapper")
        tokenizer = wrapped_tokenizer

    tdm = tf_idf_turkish(counterized_data, tokenizer)

    return tdm, vocabulary, counterized_data, text_array, tokenizer, emoji_map
