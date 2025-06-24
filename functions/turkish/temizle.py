import unicodedata
import nltk
import re
import pandas as pd

class TurkishStr(str):
    lang = 'tr'

    _case_lookup_upper = {'İ': 'i', 'I': 'ı', 'Ğ': 'ğ', 'Ş': 'ş', 'Ü': 'ü', 'Ö': 'ö', 'Ç': 'ç'}  # lookup uppercase letters
    _case_lookup_lower = {v: k for (k, v) in _case_lookup_upper.items()}

    # here we override the lower() and upper() methods
    def lower(self):
        """
        Convert Turkish string to lowercase with proper Turkish character handling.
        
        Returns:
            TurkishStr: Lowercase version of the string with Turkish-specific conversions
        """
        chars = [self._case_lookup_upper.get(c, c) for c in self]
        result = ''.join(chars).lower()
        return TurkishStr(result)

    def upper(self):
        """
        Convert Turkish string to uppercase with proper Turkish character handling.
        
        Returns:
            TurkishStr: Uppercase version of the string with Turkish-specific conversions
        """
        chars = [self._case_lookup_lower.get(c, c) for c in self]
        result = ''.join(chars).upper()
        return TurkishStr(result)


def process_text(text: str) -> str:
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
    ## TODO: Emojileri kaldırmadan veri setine dahil edebiliriz.


    metin = str(text)  # Metni string'e çevir
    secilen_kategoriler = ['Ll']
    metin = TurkishStr(metin).lower()
    zamirler = nltk.corpus.stopwords.words('turkish')
    kategoriler = [unicodedata.category(karakter) for karakter in metin]
    yeni_metin = "".join([metin[j] if kategoriler[j] in secilen_kategoriler
                          else ' ' for j in range(len(metin))])
    metin = re.sub(' +', ' ', yeni_metin)
    metin = re.sub(r'\b[xX]{2,}\b', '', metin)
    metin = [i for i in metin.split() if i not in zamirler]
    metin = ' '.join(metin)
    return metin


def metin_temizle(df: pd.DataFrame, desired_column: str) -> list:
    """
    Bu fonksiyon, verilen DataFrame'deki belirtilen sütundaki metinleri temizler.
    Temizleme işlemi, metindeki özel karakterleri ve sayıları kaldırmayı içerir.

    Args:
        df (pd.DataFrame): İşlenecek DataFrame.
        desired_column (str): Temizlenecek metin sütununun adı.

    Returns:
        pd.DataFrame: Temizlenmiş metinleri içeren DataFrame.
    """

    metin = [process_text(i) for i in df[desired_column].values]

    return metin
