import unicodedata
import nltk
import re
import pandas as pd
import emoji.core as emoji
from ...utils.combine_number_suffix import remove_space_between_terms


class TurkishStr(str):
    lang = 'tr'

    _case_lookup_upper = {'İ': 'i', 'I': 'ı', 'Ğ': 'ğ', 'Ş': 'ş', 'Ü': 'ü', 'Ö': 'ö', 'Ç': 'ç'}  # lookup uppercase letters
    _case_lookup_lower = {v: k for (k, v) in _case_lookup_upper.items()}

    # here we override the lower() and upper() methods
    def lower(self):
        chars = [self._case_lookup_upper.get(c, c) for c in self]
        result = ''.join(chars).lower()
        return TurkishStr(result)

    def upper(self):
        chars = [self._case_lookup_lower.get(c, c) for c in self]
        result = ''.join(chars).upper()
        return TurkishStr(result)


def process_text(text: str, emoji_map=None) -> str:
    """
    Temizleme işlemi, metindeki özel karakterleri ve sayıları kaldırmayı içerir.
    Also removes stopwords.
    Also removes double letters.
    Also removes extra spaces.
    Takes a string and returns a string.

    Args:
        text (str): Temizlenecek metin.

    Returns:
        str: Temizlenmiş metin.
    """


    if emoji.emoji_count(text) > 0:
        if emoji_map is not False and emoji_map is not None:
            text = emoji_map.process_text(text)
        else:
            text = emoji.replace_emoji(text, replace='emoji')


    metin = str(text)  # Metni string'e çevir
    secilen_kategoriler = ['Ll',"Nd"]
    metin = TurkishStr(metin).lower()
    zamirler = nltk.corpus.stopwords.words('turkish')
    kategoriler = [unicodedata.category(karakter) for karakter in metin]
    yeni_metin = "".join([metin[j] if kategoriler[j] in secilen_kategoriler
                          else ' ' for j in range(len(metin))])
    metin = re.sub(' +', ' ', yeni_metin)
    metin = re.sub(r'\b[xX]{2,}\b', '', metin)
    # remove repeated letters (only if 3 or more repetitions)
    metin = re.sub(r'(.)\1{2,}', r'\1', metin)
    metin = [i for i in metin.split() if i not in zamirler]
    metin = ' '.join(metin)
    metin = remove_space_between_terms(metin, r"\d+", "gb", "next")
    metin = remove_space_between_terms(metin, r"\d+", "tl", "next")
    metin = remove_space_between_terms(metin, r"\d+", "saniye", "next")
    metin = remove_space_between_terms(metin, r"\d+", "sn", "next")
    return metin


def metin_temizle_turkish(df: pd.DataFrame, desired_column: str, emoji_map=None) -> list:
    """
    Bu fonksiyon, verilen DataFrame'deki belirtilen sütundaki metinleri temizler.
    Temizleme işlemi, metindeki özel karakterleri ve sayıları kaldırmayı içerir.

    Args:
        df (pd.DataFrame): İşlenecek DataFrame.
        desired_column (str): Temizlenecek metin sütununun adı.

    Returns:
        pd.DataFrame: Temizlenmiş metinleri içeren DataFrame.
    """

    metin = [process_text(i, emoji_map) for i in df[desired_column].values]

    return metin
