from typing import List
from collections import Counter



def create_english_vocab(cleaned_data: List[str], alanadi: str, lemmatize=False, emoji_map=None) -> tuple:
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
        clean_english_text() which handle tokenization, lemmatization, and other
        text cleaning steps.
    """
    if lemmatize:
            print("Lemmatization is enabled")
    else:
        print("Lemmatization is disabled")
    # Use Counter for efficient single-pass vocabulary building
    word_counter = Counter()
    for doc in cleaned_data:
        word_counter.update(doc.split())

    # Extract unique words and sort them
    sozluk = sorted(word_counter.keys())

    return sozluk, len(cleaned_data)

    # def sozluk_goster(self):
    #     return self.sozluk
    #
    # def sozluk_sil(self, key):
    #     del self.sozluk[key]
    #
    # def sozluk_ara(self, key):
    #     return self.sozluk[key]
    #
    # def sozluk_guncelle(self, key, value):
    #     self.sozluk[key] = value
    #
    # def sozluk_uzunlugu(self):
    #     return len(self.sozluk)
    #
    # def sozluk_temizle(self):
    #     self.sozluk.clear()
