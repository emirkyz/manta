import time

from manta._functions.english.english_preprocessor import clean_english_text
from manta._functions.english.english_text_encoder import counterize_english
from manta._functions.english.english_vocabulary import create_english_vocab
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
            - vocab (dict): Vocabulary dictionary mapping words to indices
            - counterized_data (scipy.sparse matrix): TF-IDF transformed numerical data

    Raises:
        KeyError: If desired_columns is not found in the DataFrame
        ValueError: If the DataFrame is empty or contains no valid text data
    """
    text_array = clean_english_text(metin=df[desired_columns], lemmatize=lemmatize, emoji_map=emoji_map)
    print(f"Preprocess completed in {time.time() - START_TIME:.2f} seconds")
    vocab, N = create_english_vocab(text_array, desired_columns, lemmatize=lemmatize)
    counterized_data = counterize_english(N, vocab=vocab, data=text_array, field_name=desired_columns, lemmatize=lemmatize)
    # tfidf
    tdm = tf_idf_english(N, vocab=vocab, data=counterized_data, fieldname=desired_columns, output_dir=None,
                         lemmatize=lemmatize)

    return tdm, vocab, counterized_data, text_array, emoji_map
