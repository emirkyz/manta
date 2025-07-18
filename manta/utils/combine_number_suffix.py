import re


def remove_number_suffix_space_OLD(sentence, suffix):
    """
    Remove space between a number and specified suffix in a sentence.

    Args:
        sentence (str): The input sentence to process
        suffix (str): The suffix to look for after numbers

    Returns:
        str: The sentence with spaces removed between numbers and the suffix

    Examples:
        >>> remove_number_suffix_space("I have 1 gb of storage", "gb")
        'I have 1gb of storage'
        >>> remove_number_suffix_space("It costs 50 tl per item", "tl")
        'It costs 50tl per item'
        >>> remove_number_suffix_space("Buy 2 kg apples and 5 kg oranges", "kg")
        'Buy 2kg apples and 5kg oranges'
    """
    # Create regex pattern to match: number + space + suffix
    # \d+ matches one or more digits
    # \s+ matches one or more whitespace characters
    # The suffix is escaped to handle special regex characters
    pattern = r'(\d+)\s+(' + re.escape(suffix) + r')'

    # Replace with number directly followed by suffix (no space)
    result = re.sub(pattern, r'\1\2', sentence)

    return result

def remove_space_between_terms(sentence, search_term, suffix, direction="next"):
    """
    Remove space between a search term and suffix in a sentence.

    Args:
        sentence (str): The input sentence to process
        search_term (str): The search term to look for (can be number, word, or unicode)
        suffix (str): The suffix to look for after the search term
        direction (str): "next" to look for suffix after search_term,
                        "prev" to look for search_term after suffix

    Returns:
        str: The sentence with spaces removed between search_term and suffix

    Examples:
        >>> remove_space_between_terms("I have 1 gb of storage", r"\d+", "gb", "next")
        'I have 1gb of storage'
        >>> remove_space_between_terms("It costs 50 tl per item", r"\d+", "tl", "next")
        'It costs 50tl per item'
        >>> remove_space_between_terms("The gb 500 drive", "gb", r"\d+", "next")
        'The gb500 drive'
        >>> remove_space_between_terms("Price € 100 total", "€", r"\d+", "next")
        'Price €100 total'
    """
    # Escape special regex characters in suffix if it's not already a regex pattern
    if not (suffix.startswith('\\') or '[' in suffix or '(' in suffix):
        suffix_escaped = re.escape(suffix)
    else:
        suffix_escaped = suffix

    # Escape special regex characters in search_term if it's not already a regex pattern
    if not (search_term.startswith('\\') or '[' in search_term or '(' in search_term):
        search_escaped = re.escape(search_term)
    else:
        search_escaped = search_term

    if direction == "next":
        # Pattern: search_term + space + suffix
        pattern = r'(' + search_escaped + r')\s+(' + suffix_escaped + r')'
    else:  # direction == "prev"
        # Pattern: suffix + space + search_term
        pattern = r'(' + suffix_escaped + r')\s+(' + search_escaped + r')'

    # Replace with terms directly connected (no space)
    result = re.sub(pattern, r'\1\2', sentence)

    return result

test = "I have 1 of gb of storage and 50 tl in my wallet"

if __name__ == "__main__":
    print(remove_space_between_terms(test, r"\d+", "gb", "next"))