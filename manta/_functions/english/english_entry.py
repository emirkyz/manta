import time

from manta._functions.english.english_preprocessor import metin_temizle_english
from manta._functions.english.english_text_encoder import sayisallastirma
from manta._functions.english.english_vocabulary import sozluk_yarat
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

    return tdm, sozluk, sayisal_veri, metin_array,emoji_map
