from functions.english.process import preprocess

 
def sayisallastirma(N, sozluk, data, alanadi, lemmatize,tokenizer):
    """
    Convert text documents to numerical representation using tokenizer encoding.
    
    Args:
        N: Number parameter (unused in current implementation)
        sozluk: Dictionary/vocabulary (unused in current implementation)
        data (DataFrame): DataFrame containing text data
        alanadi (str): Column name containing text data
        lemmatize (bool): Whether to apply lemmatization during preprocessing
        tokenizer: Tokenizer object with encode method
    
    Returns:
        list or None: List of tokenized document IDs or None on error
    """
    sayisal_veri = []
    dokumanlar = data[alanadi].tolist()
    #update_progress_emit(50, "TF-IDF Hesaplanıyor", "PROCESSING", "tfidf", "tid")
    try:
        for i in range(len(data)):
            kelime = preprocess(dokumanlar[i], lemmatize=lemmatize)
            # convert to str
            str_document = " ".join(kelime)
            sayisal_veri.append(tokenizer.encode(str_document).ids)
        return sayisal_veri

    except Exception as e:
        print(f"Error in sayisallastirma: {e}")
        return None
