import re
import unicodedata
from typing import List, Any

import emoji
import nltk
import functools

# Module-level initialization for better performance
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
LEMMATIZER = nltk.stem.WordNetLemmatizer()
STEMMER = nltk.stem.SnowballStemmer('english')

# Precompiled regex patterns
WHITESPACE_PATTERN = re.compile(r' +')
XXX_PATTERN = re.compile(r'\b[xX]{2,}\b')


@functools.cache
def preprocess(metin=None, lemmatize=False, kategoriler=frozenset(),emoji_map=None) -> str:
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
    # Use module-level stemmer/lemmatizer

    if "1. Very small left apical pneumothorax. 2. Atelectasis at the base of the left lung. _ " in metin:
        pass
    if lemmatize:
        budayici = LEMMATIZER
    else:
        budayici = STEMMER

    if emoji.emoji_count(metin) > 0:
        if emoji_map is not None:
            metin = emoji_map.process_text(metin)
        else:
            metin = emoji.replace_emoji(metin, replace='emoji')

    if metin is None:
        return []
    
    metin = metin.lower()
    metin = unicodedata.normalize('NFKD', metin)
    
    # Optimize Unicode character filtering
    secilen_kategoriler = {'Ll', 'Nd'}
    yeni_metin = ''.join(char if unicodedata.category(char) in secilen_kategoriler else ' ' 
                        for char in metin)
    
    # Use precompiled patterns
    metin = WHITESPACE_PATTERN.sub(' ', yeni_metin)
    metin = XXX_PATTERN.sub('', metin)
    metin = metin.strip()

    metin = metin.split()
    metin = [karakter for karakter in metin if karakter not in STOPWORDS]

    if lemmatize:
        metin = [budayici.lemmatize(parca) for parca in metin]
    else:
        metin = [budayici.stem(parca) for parca in metin]

    metin = ' '.join(metin)
    return metin


def metin_temizle_english(metin=None, lemmatize=False, kategoriler=frozenset(),emoji_map=None) -> List[str]:
    """
    Preprocesses text data by applying lemmatization (if enabled) and removing stopwords.

    This function performs text normalization, including lowercasing, Unicode normalization,
    and removing specific character categories. It also removes common English stopwords
    and applies lemmatization (if enabled).
    """
    
    metin = [preprocess(metin=i, lemmatize=lemmatize, kategoriler=kategoriler, emoji_map=emoji_map) for i in metin]
    return metin
