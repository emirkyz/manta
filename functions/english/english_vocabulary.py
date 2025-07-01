import pandas as pd


def sozluk_yarat(cleaned_data: pd.DataFrame, alanadi: str,lemmatize=False,emoji_map=None) -> tuple:
    """
    Creates a vocabulary list from a DataFrame column of text data.

    This function processes text data by tokenizing it, applying lemmatization
    (if enabled), and creating a set of unique tokens. It then converts this
    set into a sorted list of tokens, which serves as a vocabulary for further
    text analysis.

    Args:
        veri (pd.DataFrame): The DataFrame containing the text data.
        alanadi (str): The name of the column in the DataFrame that contains the text data.
        lemmatize (bool, optional): Whether to apply lemmatization to the text data. Defaults to False.

    Returns:
        tuple: A tuple containing the vocabulary list and the number of documents processed.
    """
    if lemmatize:
            print("Lemmatization is enabled")
    else:
        print("Lemmatization is disabled")
    sozluk = set()
    data = cleaned_data
    for dokuman in data:
        sozluk.update(dokuman)
    
    return sorted(sozluk), len(data)

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
