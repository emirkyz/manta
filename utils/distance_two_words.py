import Levenshtein

def calc_levenstein_distance(word1, word2):
    return Levenshtein.distance(word1, word2)

def calc_cosine_distance(word1, word2):
    return Levenshtein.ratio(word1, word2)

def calc_jaccard_distance(word1, word2):
    return Levenshtein.jaro_winkler(word1, word2)