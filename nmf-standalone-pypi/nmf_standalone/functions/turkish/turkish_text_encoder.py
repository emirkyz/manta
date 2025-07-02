import pandas as pd


def veri_sayisallastir(arr, tokenizer):
    """
    Convert text data from DataFrame to numerical format using the tokenizer.
    Takes a list of text data and a trained tokenizer.
    Args:
        arr: list of text data
        tokenizer: trained Tokenizer object
    
    Returns:
        list of numerical representations of documents
    """
    # Ensure all documents are strings before tokenization
    sayisal_veri = [tokenizer.encode(str(dokuman)).ids for dokuman in arr if not pd.isna(dokuman)]
    '''for dokuman in df[desired_column].values:
        sayisal_dokuman = tokenizer.encode(dokuman).ids
        sayisal_veri.append(sayisal_dokuman)'''
    return sayisal_veri
