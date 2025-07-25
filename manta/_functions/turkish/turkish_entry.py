from .turkish_preprocessor import metin_temizle_turkish, process_texts_generator
from .turkish_tokenizer_factory import init_tokenizer, train_tokenizer, train_tokenizer_from_generator
from .turkish_text_encoder import veri_sayisallastir
from ..tfidf import tf_idf_turkish
import itertools


def process_turkish_file(df, desired_columns: str, tokenizer=None, tokenizer_type=None, emoji_map=None):
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

    Returns:
        tuple: A tuple containing:
            - tdm (scipy.sparse matrix): Term-document matrix (TF-IDF transformed)
            - sozluk (list): Vocabulary list from the tokenizer
            - sayisal_veri (scipy.sparse matrix): Numerical representation of documents
            - tokenizer: Trained tokenizer instance
            - metin_array (list): Cleaned text array
            - emoji_map (EmojiMap): Emoji mapping instance used

    Raises:
        ValueError: If tokenizer_type is not supported
        KeyError: If desired_columns is not found in the DataFrame
    """

    metin_array = metin_temizle_turkish(df, desired_columns, emoji_map=emoji_map)
    print(f"Number of documents: {len(metin_array)}")

    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = init_tokenizer(tokenizer_type=tokenizer_type)

    # Train the tokenizer
    tokenizer = train_tokenizer(tokenizer, metin_array, tokenizer_type=tokenizer_type)
    sozluk = list(tokenizer.get_vocab().keys())

    # sayısallaştır
    sayisal_veri = veri_sayisallastir(metin_array, tokenizer)
    tdm = tf_idf_turkish(sayisal_veri, tokenizer)

    return tdm, sozluk, sayisal_veri, tokenizer, metin_array, emoji_map


def process_turkish_file_from_generator(csv_generator, desired_columns: str, tokenizer=None, tokenizer_type=None, emoji_map=None):
    """
    Process Turkish text data for topic modeling using NMF with streaming data.

    This function performs text preprocessing, tokenization, and TF-IDF transformation
    specifically for Turkish language texts using generators for memory efficiency.

    Args:
        csv_generator: Generator yielding CSV row dictionaries
        desired_columns (str): Name of the column containing text to analyze
        tokenizer (optional): Pre-trained tokenizer instance. If None, a new tokenizer
                             will be initialized based on tokenizer_type
        tokenizer_type (str, optional): Type of tokenizer to use. Options: "bpe" or "wordpiece"
        emoji_map (EmojiMap, optional): Emoji mapping instance for emoji processing

    Returns:
        tuple: A tuple containing:
            - tdm (scipy.sparse matrix): Term-document matrix (TF-IDF transformed)
            - sozluk (list): Vocabulary list from the tokenizer
            - sayisal_veri (scipy.sparse matrix): Numerical representation of documents
            - tokenizer: Trained tokenizer instance
            - metin_array (list): Cleaned text array (collected from generator)
            - emoji_map (EmojiMap): Emoji mapping instance used

    Raises:
        ValueError: If tokenizer_type is not supported
        KeyError: If desired_columns is not found in the generator data
    """
    
    print("Processing Turkish file using generators...")
    
    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = init_tokenizer(tokenizer_type=tokenizer_type)

    # Create text processing generator
    text_generator = process_texts_generator(csv_generator, desired_columns, emoji_map=emoji_map)
    
    # Use itertools.tee to create two independent generators from the same source
    tokenizer_generator, text_collection_generator = itertools.tee(text_generator, 2)
    
    print("Training tokenizer from generator (streaming)...")
    # Train the tokenizer directly from generator
    tokenizer = train_tokenizer_from_generator(tokenizer, tokenizer_generator, tokenizer_type=tokenizer_type)
    sozluk = list(tokenizer.get_vocab().keys())
    
    print("Collecting texts for numerical processing...")
    # Collect texts for subsequent processing steps that still require arrays
    metin_array = list(text_collection_generator)
    
    print(f"Number of documents: {len(metin_array)}")
    
    # sayısallaştır - using original array-based encoding
    sayisal_veri = veri_sayisallastir(metin_array, tokenizer)
    tdm = tf_idf_turkish(sayisal_veri, tokenizer)

    return tdm, sozluk, sayisal_veri, tokenizer, metin_array, emoji_map
