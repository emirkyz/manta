import time

from manta._functions.english.english_preprocessor import metin_temizle_english, process_texts_generator_english
from manta._functions.english.english_text_encoder import sayisallastirma
from manta._functions.english.english_vocabulary import sozluk_yarat, sozluk_yarat_from_generator
from manta._functions.tfidf import tf_idf_english

START_TIME = time.time()


def process_english_file(df, desired_columns: str, lemmatize: bool,emoji_map=None):
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

    Returns:
        tuple: A tuple containing:
            - tdm (scipy.sparse matrix): Term-document matrix (TF-IDF transformed)
            - sozluk (dict): Vocabulary dictionary mapping words to indices
            - sayisal_veri (scipy.sparse matrix): TF-IDF transformed numerical data

    Raises:
        KeyError: If desired_columns is not found in the DataFrame
        ValueError: If the DataFrame is empty or contains no valid text data
    """
    metin_array = metin_temizle_english(metin=df[desired_columns], lemmatize=lemmatize, emoji_map=emoji_map)
    print(f"Preprocess completed in {time.time() - START_TIME:.2f} seconds")
    sozluk, N = sozluk_yarat(metin_array, desired_columns, lemmatize=lemmatize)
    sayisal_veri = sayisallastirma(N, sozluk=sozluk, data=metin_array, alanadi=desired_columns, lemmatize=lemmatize)
    # tfidf
    tdm = tf_idf_english(N, sozluk=sozluk, data=sayisal_veri, alanadi=desired_columns, output_dir=None,
                                 lemmatize=lemmatize)

    return tdm, sozluk, sayisal_veri, metin_array


def process_english_file_from_generator(csv_generator, desired_columns: str, lemmatize: bool, emoji_map=None):
    """
    Process English text data for topic modeling using NMF with streaming data.

    This function performs text preprocessing and TF-IDF transformation specifically
    for English language texts using generators for memory efficiency.

    Args:
        csv_generator: Generator yielding CSV row dictionaries
        desired_columns (str): Name of the column containing text to analyze
        lemmatize (bool): Whether to apply lemmatization during preprocessing.
                         If True, words are reduced to their base forms
        emoji_map: Optional emoji mapping

    Returns:
        tuple: A tuple containing:
            - tdm (scipy.sparse matrix): Term-document matrix (TF-IDF transformed)
            - sozluk (dict): Vocabulary dictionary mapping words to indices
            - sayisal_veri (scipy.sparse matrix): TF-IDF transformed numerical data
            - metin_array (list): Cleaned text array (collected from generator)

    Raises:
        KeyError: If desired_columns is not found in the generator data
        ValueError: If no valid text data is found
    """
    print("Processing English file using generators...")
    
    # Create text processing generator
    text_generator = process_texts_generator_english(csv_generator, desired_columns, lemmatize=lemmatize, emoji_map=emoji_map)
    
    # Use itertools.tee to create two independent generators from the same source
    import itertools
    vocab_generator, text_collection_generator = itertools.tee(text_generator, 2)
    
    print("Creating vocabulary from generator (streaming)...")
    # Create vocabulary directly from generator without collecting texts
    sozluk, N = sozluk_yarat_from_generator(vocab_generator, desired_columns, lemmatize=lemmatize)
    
    print("Collecting texts for numerical processing...")
    # Collect texts for subsequent processing steps that still require arrays
    metin_array = list(text_collection_generator)
    
    print(f"Preprocess completed in {time.time() - START_TIME:.2f} seconds")
    print(f"Number of documents: {N}")
    
    # Create numerical representation - using original array-based encoding
    sayisal_veri = sayisallastirma(N, sozluk=sozluk, data=metin_array, alanadi=desired_columns, lemmatize=lemmatize)
    
    # Apply TF-IDF transformation
    tdm = tf_idf_english(N, sozluk=sozluk, data=sayisal_veri, alanadi=desired_columns, output_dir=None,
                         lemmatize=lemmatize)

    return tdm, sozluk, sayisal_veri, metin_array
