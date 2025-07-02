# Set tokenizers parallelism before importing tokenizers


from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def init_tokenizer(tokenizer_type: str = "wordpiece"):
    """
    Initialize a new WordPiece tokenizer with default settings.
    This should be called once at the start of the program.
    
    Returns:
        Tokenizer: An initialized but untrained tokenizer
    """
    if tokenizer_type == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[BİLİNMİYOR]"))
        print("Initialized BPE tokenizer")
    elif tokenizer_type == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token="[BİLİNMİYOR]"))
        print("Initialized Wordpiece tokenizer")
    else:
        raise ValueError(f"Invalid tokenizer type: {tokenizer_type}")
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def train_tokenizer(tokenizer: Tokenizer, arr: list, tokenizer_type: str = "wordpiece"):
    """
    Train an initialized tokenizer on the specified text data.
    
    Args:
        tokenizer: An initialized Tokenizer object
        arr: list of text data
    
    Returns:
        Tokenizer: The trained tokenizer
    """
    if tokenizer_type == "bpe":
        trainer = BpeTrainer()
        print("Initialized BPE trainer")
    elif tokenizer_type == "wordpiece":
        trainer = WordPieceTrainer(vocab_size=128* 1024,
                                   min_frequency=5,
                                   special_tokens=["[BİLİNMİYOR]"])
        print("Initialized WordPiece trainer")
    tokenizer.train_from_iterator(arr, trainer)
    return tokenizer


def gen_token(arr, tokenizer_type: str = "wordpiece"):
    """
    Generate a WordPiece tokenizer from the text in the specified DataFrame column.
    For backward compatibility - prefer using init_tokenizer() and train_tokenizer() separately.
    
    Args:
        arr: list of text data
    
    Returns:
        trained Tokenizer object
    """
    tokenizer = init_tokenizer(tokenizer_type)
    return train_tokenizer(tokenizer, arr, tokenizer_type)
