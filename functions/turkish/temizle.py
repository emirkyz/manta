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

    if text == "TÜM BİP UYGULAMALARI HATALARLA DOLU. SÖZDE HEDİYELER VERİP MİLLETİ CEZBETMEYE ÇALIŞIYORLAR AMA BEŞ KURUŞ ETMEZ Bİ UYGULAMA. SİZ BU GİDİŞLE DEĞİL WHATSAPPLA UYDURUK PROGRAMLARLA BİLE BOY ÖLÇÜŞEMEZSİNİZ. BİPTEN HER SANİYE ARAMA YAP PUAN KAZAN DİYORLAR. Bİ TON ARIYORSUN PUAN FALAN ORTADA YOK. LİFECELLE HER AYIN 14 ü 14:40 ta sözde hediye veriyor. AMA HAK GETİRE KISACA BİP BÜYÜK BİR FİYASKO... FAZLADAN 1 2GB İÇİN TELEFONUNUZDA YER KAPLAMASINA VE KULLANMAYA DEĞMEZ BİLE...":

        print("hmmmm")

    metin = str(text)  # Metni string'e çevir
    secilen_kategoriler = ['Ll',"Mn"]
    metin = metin.lower()
    zamirler = nltk.corpus.stopwords.words('turkish')
    metin = unicodedata.normalize('NFD', metin)
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
