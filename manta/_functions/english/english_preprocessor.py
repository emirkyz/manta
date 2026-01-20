import re
import unicodedata
from typing import List
import multiprocessing as mp
import os

import emoji.core as emoji
import nltk
import functools

# Module-level initialization for better performance
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
LEMMATIZER = nltk.stem.WordNetLemmatizer()
STEMMER = nltk.stem.SnowballStemmer('english')

# Precompiled regex patterns for better performance
WHITESPACE_PATTERN = re.compile(r' +')
XXX_PATTERN = re.compile(r'\b[xX]{2,}\b')
EMOJI_PATTERN = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
SELECTED_CATEGORIES = frozenset(['Ll'])
SELECTED_CATEGORIES_WITH_NUMBERS = frozenset(['Ll', 'Nd'])

def has_emoji_fast(text: str) -> bool:
    """Fast emoji detection using regex pattern."""
    return bool(EMOJI_PATTERN.search(text))


@functools.cache
def preprocess(text=None, lemmatize=False, categories=frozenset(), emoji_map=None, keep_numbers=False) -> str:
    """
    Preprocesses text data by applying lemmatization (if enabled) and removing stopwords.

    This function performs text normalization, including lowercasing, Unicode normalization,
    and removing specific character categories. It also removes common English stopwords
    and applies lemmatization (if enabled).

    Args:
        text (str): The text data to be preprocessed.
        lemmatize (bool): Whether to apply lemmatization to the text data.
        categories (frozenset): A set of character categories to be removed.
        keep_numbers (bool): If True, preserve numeric characters (Nd category) for BPE merging.

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
    
    # Optimized emoji processing - use fast detection first
    if has_emoji_fast(text):
        if emoji_map is not False and emoji_map is not None:
            text = emoji_map.process_text(text)
        else:
            text = emoji.replace_emoji(text, replace='emoji')

    if text is None:
        return []

    text = text.lower()
    text = unicodedata.normalize('NFKD', text)

    # Select categories based on keep_numbers flag
    active_categories = SELECTED_CATEGORIES_WITH_NUMBERS if keep_numbers else SELECTED_CATEGORIES

    # Optimize Unicode character filtering using selected categories
    yeni_metin = ''.join(char if unicodedata.category(char) in active_categories else ' '
                         for char in text)

    # Use precompiled patterns
    text = WHITESPACE_PATTERN.sub(' ', yeni_metin)
    text = XXX_PATTERN.sub('', text)
    text = text.strip()

    # Combine stopword filtering with stemming/lemmatization in single pass
    if lemmatize:
        text = [budayici.lemmatize(word) for word in text.split() if word not in STOPWORDS]
    else:
        text = [budayici.stem(word) for word in text.split() if word not in STOPWORDS]

    # Join with space
    text = ' '.join(text)
    return text


def process_batch_nltk(words_batch: List[str], lemmatize: bool) -> List[str]:
    """Vectorized NLTK processing for better performance."""
    if lemmatize:
        return [LEMMATIZER.lemmatize(word) for word in words_batch]
    else:
        return [STEMMER.stem(word) for word in words_batch]

def clean_english_text(metin=None, lemmatize=False, kategoriler=frozenset(), emoji_map=None, keep_numbers=False) -> List[str]:
    """
    Preprocesses text data by applying lemmatization (if enabled) and removing stopwords.

    This function performs text normalization, including lowercasing, Unicode normalization,
    and removing specific character categories. It also removes common English stopwords
    and applies lemmatization (if enabled).

    Args:
        metin: Array of text data to preprocess.
        lemmatize (bool): Whether to apply lemmatization.
        kategoriler (frozenset): Character categories to keep.
        emoji_map: Emoji mapping for emoji processing.
        keep_numbers (bool): If True, preserve numeric characters for BPE merging.

    Returns:
        List[str]: Preprocessed text data.
    """
    # Check if parallel processing would be beneficial (for large datasets)
    if len(metin) > 1000:
        # Use multiprocessing for large datasets
        cpu_count = min(os.cpu_count() or 1, 4)  # Limit to 4 processes max
        chunk_size = max(1, len(metin) // cpu_count)

        try:
            from joblib import Parallel, delayed

            #metin = Parallel(n_jobs=cpu_count, prefer="threads", batch_size="auto")(
            #    delayed(preprocess)(text, lemmatize=lemmatize, categories=kategoriler,
            #                        emoji_map=emoji_map, keep_numbers=keep_numbers)
            #    for text in metin
            #)
            # Use fork context - no module re-import issue
            ctx = mp.get_context('fork')
            with ctx.Pool(processes=cpu_count) as pool:
                process_func = functools.partial(preprocess, lemmatize=lemmatize,
                                                 categories=kategoriler, emoji_map=emoji_map,
                                                 keep_numbers=keep_numbers)
                metin = pool.map(process_func, metin, chunksize=chunk_size)

        except Exception:
            # Fall back to sequential processing if parallel fails
            metin = [preprocess(text=i, lemmatize=lemmatize, categories=kategoriler, emoji_map=emoji_map, keep_numbers=keep_numbers) for i in metin]
    else:
        # Use sequential processing for smaller datasets
        metin = [preprocess(text=i, lemmatize=lemmatize, categories=kategoriler, emoji_map=emoji_map, keep_numbers=keep_numbers) for i in metin]

    return metin
