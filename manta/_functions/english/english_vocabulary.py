from typing import List

import pandas as pd


def sozluk_yarat(cleaned_data: List[str], alanadi: str,lemmatize=False,emoji_map=None) -> tuple:
    """
    Creates a vocabulary list from preprocessed text data.

    This function takes preprocessed text data and creates a vocabulary by extracting unique tokens
    from all documents. The tokens are sorted alphabetically to create a consistent vocabulary order.

    Args:
        cleaned_data (List[str]): List of preprocessed text documents
        alanadi (str): Name of the field/column containing the text data (used for logging)
        lemmatize (bool, optional): Whether lemmatization was applied during preprocessing. 
                                  Defaults to False.
        emoji_map (EmojiMap, optional): Emoji mapping instance used during preprocessing.
                                      Defaults to None.

    Returns:
        tuple: A tuple containing:
            - list: Sorted vocabulary list of unique tokens
            - int: Number of documents processed

    Note:
        The input cleaned_data should already be preprocessed using functions like
        metin_temizle_english() which handle tokenization, lemmatization, and other
        text cleaning steps.
    """
    if lemmatize:
            print("Lemmatization is enabled")
    else:
        print("Lemmatization is disabled")
    # Process all documents in a single pass using set comprehension
    sozluk = set().union(*[set(doc.split()) for doc in cleaned_data])
    
    return sorted(sozluk), len(cleaned_data)


def sozluk_yarat_from_generator(text_generator, alanadi: str, lemmatize=False, emoji_map=None) -> tuple:
    """
    Creates a vocabulary list from preprocessed text data using a generator (streaming).

    This function takes a generator yielding preprocessed text data and creates a vocabulary by 
    extracting unique tokens incrementally. Memory efficient for large datasets.

    Args:
        text_generator: Generator yielding preprocessed text strings
        alanadi (str): Name of the field/column containing the text data (used for logging)
        lemmatize (bool, optional): Whether lemmatization was applied during preprocessing. 
                                  Defaults to False.
        emoji_map (EmojiMap, optional): Emoji mapping instance used during preprocessing.
                                      Defaults to None.

    Returns:
        tuple: A tuple containing:
            - list: Sorted vocabulary list of unique tokens
            - int: Number of documents processed

    Note:
        The input text_generator should yield already preprocessed text strings.
    """
    if lemmatize:
        print("Lemmatization is enabled")
    else:
        print("Lemmatization is disabled")
    
    print(f"Creating vocabulary from generator for column: {alanadi}")
    
    vocab_set = set()
    doc_count = 0
    
    for text in text_generator:
        if text and text.strip():  # Skip empty texts
            # Add words from this document to vocabulary
            words = text.split()
            vocab_set.update(words)
            doc_count += 1

    
    print(f"Vocabulary creation completed. Total documents: {doc_count}, vocabulary size: {len(vocab_set)}")
    return sorted(vocab_set), doc_count

