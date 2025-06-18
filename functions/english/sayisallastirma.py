from functions.english.process import preprocess

 
def sayisallastirma(N, sozluk, data, alanadi, lemmatize,tokenizer):
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
