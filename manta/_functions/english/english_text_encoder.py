from .english_preprocessor import preprocess

 
def counterize_english(N, vocab, data, field_name, lemmatize):
    """
    Convert text documents to numerical representation using vocabulary indices.
    
    This function takes preprocessed text documents and converts them to numerical form by mapping
    each word to its index in the vocabulary. Words not found in the vocabulary are mapped to 0.
    The vocabulary should be pre-sorted and consistent with the one used during preprocessing.
    
    Args:
        N (int): Total number of documents in the dataset. Used for validation.
        vocab (list): Sorted vocabulary list where each word corresponds to a unique index.
                       The vocabulary should match the one used during preprocessing.
        data (list): List of preprocessed text documents, where each document is a space-separated string.
        field_name (str): Field/column name in data containing the document texts. Currently unused.
        lemmatize (bool): Whether lemmatization was applied during preprocessing. Currently unused.
                         Should match the setting used during preprocessing.

    Returns:
        list: List of lists where each inner list contains integer indices corresponding to words
              in the document. The indices match positions in the vocabulary list. Out-of-vocabulary
              words are mapped to 0.

    Note:
        The input data should already be preprocessed using functions like clean_english_text()
        and use the same preprocessing parameters (lemmatization, etc.) as when creating the vocabulary.
    """
    numerical_data = []
    documents = data
    
    # Create vocabulary mapping from word to index
    vocab_to_index = {word: idx for idx, word in enumerate(vocab)}
    
    # Encode each document
    for document in documents:
        # Split document into words
        words = document.split()
        # Convert words to indices, use 0 for unknown words (out-of-vocabulary)
        document_indices = [vocab_to_index.get(word, 0) for word in words]
        numerical_data.append(document_indices)
    
    return numerical_data
    