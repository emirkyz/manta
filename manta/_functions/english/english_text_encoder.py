import pandas as pd


def veri_sayisallastir(arr, tokenizer):
    """
    Convert text data from DataFrame to numerical format using the tokenizer.
    Takes a list of text data and a trained tokenizer.
    Args:
        arr: list of text data
        tokenizer: trained Tokenizer object
    
    Returns:
        list of numerical representations of documents
    """
    # Ensure all documents are strings before tokenization
    sayisal_veri = [tokenizer.encode(str(dokuman)).ids for dokuman in arr if not pd.isna(dokuman)]
    return sayisal_veri


def sayisallastirma(N, sozluk, data, alanadi, lemmatize):
    """
    DEPRECATED: Convert text documents to numerical representation using vocabulary indices.
    
    This function is kept for backward compatibility but is deprecated.
    Use veri_sayisallastir() with tokenizers instead.
    
    This function takes preprocessed text documents and converts them to numerical form by mapping
    each word to its index in the vocabulary. Words not found in the vocabulary are mapped to 0.
    The vocabulary should be pre-sorted and consistent with the one used during preprocessing.
    
    Args:
        N (int): Total number of documents in the dataset. Used for validation.
        sozluk (list): Sorted vocabulary list where each word corresponds to a unique index.
                       The vocabulary should match the one used during preprocessing.
        data (list): List of preprocessed text documents, where each document is a space-separated string.
        alanadi (str): Field/column name in data containing the document texts. Currently unused.
        lemmatize (bool): Whether lemmatization was applied during preprocessing. Currently unused.
                         Should match the setting used during preprocessing.

    Returns:
        list: List of lists where each inner list contains integer indices corresponding to words
              in the document. The indices match positions in the vocabulary list. Out-of-vocabulary
              words are mapped to 0.

    Note:
        The input data should already be preprocessed using functions like metin_temizle_english()
        and use the same preprocessing parameters (lemmatization, etc.) as when creating the vocabulary.
    """
    sayisal_veri = []
    dokumanlar = data
    
    # Create vocabulary mapping from word to index
    vocab_to_index = {word: idx for idx, word in enumerate(sozluk)}
    
    # Encode each document
    for dokuman in dokumanlar:
        # Split document into words
        words = dokuman.split()
        # Convert words to indices, use 0 for unknown words (out-of-vocabulary)
        document_indices = [vocab_to_index.get(word, 0) for word in words]
        sayisal_veri.append(document_indices)
    
    return sayisal_veri
    