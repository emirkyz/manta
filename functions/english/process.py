import re
import unicodedata
from typing import List

import nltk
import functools


@functools.cache
def preprocess(metin=None, lemmatize=False, kategoriler=frozenset()) -> List[str]:
    """
    Preprocesses text data by applying lemmatization (if enabled) and removing stopwords.

    This function performs text normalization, including lowercasing, Unicode normalization,
    and removing specific character categories. It also removes common English stopwords
    and applies lemmatization (if enabled).

    Args:
        metin (str): The text data to be preprocessed.
        lemmatize (bool): Whether to apply lemmatization to the text data.
        kategoriler (frozenset): A set of character categories to be removed.

    Returns:
        List[str]: A list of preprocessed words.

    Raises:
        ValueError: If the input text is None.
    """
    # we can use lemmetizer as well
    if lemmatize:
        budayici = nltk.stem.WordNetLemmatizer()
    else:
        budayici = nltk.stem.SnowballStemmer('english')

    zamirler = nltk.corpus.stopwords.words('english')

    # print(self.metin)
    if metin is None:
        return []
    metin = metin.lower()
    metin = unicodedata.normalize('NFKD', metin)
    secilen_kategoriler = ['Ll', 'Zs']
    kategoriler = [unicodedata.category(karakter) for karakter in metin]
    yeni_metin = "".join([metin[j] if kategoriler[j] in secilen_kategoriler and kategoriler[j] != 'Zs'
                          else ' ' for j in range(len(metin))])
    metin = re.sub(' +', ' ', yeni_metin)
    metin = re.sub(r'\b[xX]{2,}\b', '', metin)
    metin = metin.strip()

    metin = metin.split()
    metin = [karakter for karakter in metin if karakter not in zamirler]

    if lemmatize:
        metin = [budayici.lemmatize(parca) for parca in metin]
    else:
        metin = [budayici.stem(parca) for parca in metin]


    return metin
