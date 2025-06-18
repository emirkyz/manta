import unicodedata
import nltk
import re
import pandas as pd


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

    if text == "YORUM ÜSTTE GOZUKSUN DİYE 5 YİLDİZ VERDİM TEK KELİMEYLE BERBAT ÇOK CİMRİ BİR OPERATÖR VE UYGULAMA OLMUŞ ESKİDEN İNTERNET VERİRDİ ARTİK YOUTUBE VE İNSTAGRAM VEYA TİKTOK VERİYOR ALLAH BİRAKMASİN":

        print("hmmmm")

    metin = str(text)  # Metni string'e çevir
    secilen_kategoriler = ['Ll']
    metin = metin.lower()
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
