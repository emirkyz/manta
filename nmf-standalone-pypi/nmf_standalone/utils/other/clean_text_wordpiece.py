import unicodedata
import re
import nltk
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

def clean_text(metin: str, remove_stopwords: bool = True) -> str:
    if metin is None:
        return ""
    
    # Basic cleaning
    text = metin.lower()
    text = unicodedata.normalize('NFKD', text)
    
    # Remove unwanted characters
    secilen_kategoriler = ['Ll', 'Zs']
    kategoriler = [unicodedata.category(karakter) for karakter in text]
    yeni_metin = "".join([text[j] if kategoriler[j] in secilen_kategoriler and kategoriler[j] != 'Zs'
                          else ' ' for j in range(len(text))])
    text = re.sub(' +', ' ', yeni_metin)
    text = re.sub(r'\b[xX]{2,}\b', '', text)
    text = text.strip()

    if remove_stopwords:
        zamirler = nltk.corpus.stopwords.words('turkish')
        words = text.split()
        text = ' '.join([word for word in words if word not in zamirler])

    return text


def train_wordpiece_tokenizer(documents: List[str], vocab_size: int = 8000, min_frequency: int = 5):
    # Clean all documents first
    cleaned_documents = [clean_text(doc) for doc in documents]
    
    tokenizer = Tokenizer(WordPiece(unk_token="[BILINMIYOR]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[BILINMIYOR]"],
        min_frequency=min_frequency
    )
    
    # Train on cleaned documents
    tokenizer.train_from_iterator(cleaned_documents, trainer=trainer)
    
    return tokenizer