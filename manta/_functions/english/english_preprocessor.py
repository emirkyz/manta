import re
import unicodedata
from typing import List

import emoji.core as emoji
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
def preprocess(text=None, lemmatize=False, categories=frozenset(), emoji_map=None) -> str:
    """
    Preprocesses text data by applying lemmatization (if enabled) and removing stopwords.

    This function performs text normalization, including lowercasing, Unicode normalization,
    and removing specific character categories. It also removes common English stopwords
    and applies lemmatization (if enabled).

    Args:
        text (str): The text data to be preprocessed.
        lemmatize (bool): Whether to apply lemmatization to the text data.
        categories (frozenset): A set of character categories to be removed.

    Returns:
        List[str]: A list of preprocessed words.

    Raises:
        ValueError: If the input text is None.
    """
    # Use module-level stemmer/lemmatizer

        
    if lemmatize:
        budayici = LEMMATIZER
    else:
        budayici = STEMMER
    
    if emoji.emoji_count(text) > 0:
        if emoji_map is not False and emoji_map is not None:
            text = emoji_map.process_text(text)
        else:
            text = emoji.replace_emoji(text, replace='emoji')

    if text is None:
        return []

    text = text.lower()
    text = unicodedata.normalize('NFKD', text)

    # Optimize Unicode character filtering
    secilen_kategoriler = ['Ll']
    yeni_metin = ''.join(char if unicodedata.category(char) in secilen_kategoriler else ' '
                         for char in text)

    # Use precompiled patterns
    text = WHITESPACE_PATTERN.sub(' ', yeni_metin)
    text = XXX_PATTERN.sub('', text)
    text = text.strip()

    # Split and filter stopwords in one pass
    text = [word for word in text.split() if word not in STOPWORDS]

    # Process words in bulk using map() instead of list comprehension
    if lemmatize:
        text = list(map(budayici.lemmatize, text))
    else:
        text = list(map(budayici.stem, text))

    # Join with space
    text = ' '.join(text)
    return text


def clean_english_text(metin=None, lemmatize=False, kategoriler=frozenset(), emoji_map=None) -> List[str]:
    """
    Preprocesses text data by applying lemmatization (if enabled) and removing stopwords.

    This function performs text normalization, including lowercasing, Unicode normalization,
    and removing specific character categories. It also removes common English stopwords
    and applies lemmatization (if enabled).
    """

    metin = [preprocess(text=i, lemmatize=lemmatize, categories=kategoriler, emoji_map=emoji_map) for i in metin]
    return metin
