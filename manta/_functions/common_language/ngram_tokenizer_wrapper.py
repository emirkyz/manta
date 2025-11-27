"""
Tokenizer wrapper that extends original tokenizer functionality with n-gram support.

This wrapper maintains compatibility with the original tokenizer interface
while adding support for n-gram tokens created by WordPairBPE.
"""

from typing import Dict, List, Optional, Union


class NgramTokenizerWrapper:
    """
    Wrapper around original tokenizer that adds n-gram token support.

    This class maintains the same interface as the original tokenizer
    but extends it to handle n-gram tokens created by WordPairBPE.
    """

    def __init__(self, original_tokenizer, ngram_bpe=None, extended_vocabulary=None):
        """
        Initialize the wrapper.

        Args:
            original_tokenizer: The original BPE/WordPiece tokenizer
            ngram_bpe: WordPairBPE instance that created n-grams
            extended_vocabulary: Extended vocabulary list including n-grams
        """
        self.original_tokenizer = original_tokenizer
        self.ngram_bpe = ngram_bpe
        self.extended_vocabulary = extended_vocabulary or []

        # Cache original vocab info
        self.original_vocab = original_tokenizer.get_vocab()
        self.original_vocab_size = len(self.original_vocab)

        # Build extended vocab mapping if we have n-grams
        self.extended_vocab_dict = {}
        if extended_vocabulary:
            self.extended_vocab_dict = {token: idx for idx, token in enumerate(extended_vocabulary)}

    def id_to_token(self, token_id: int) -> str:
        """
        Convert token ID to token string, supporting both original and n-gram tokens.

        Args:
            token_id: Token ID to convert

        Returns:
            Token string representation
        """
        if token_id < self.original_vocab_size:
            # This is an original token
            return self.original_tokenizer.id_to_token(token_id)
        else:
            # This is an n-gram token
            if token_id < len(self.extended_vocabulary):
                return self.extended_vocabulary[token_id]
            else:
                return f"<UNK_NGRAM_{token_id}>"

    def get_vocab(self) -> Dict[str, int]:
        """
        Get extended vocabulary dictionary including n-grams.

        Returns:
            Dictionary mapping tokens to IDs
        """
        if self.extended_vocab_dict:
            return self.extended_vocab_dict
        else:
            return self.original_vocab

    def get_vocab_size(self) -> int:
        """
        Get total vocabulary size including n-grams.

        Returns:
            Total vocabulary size
        """
        if self.extended_vocabulary:
            return len(self.extended_vocabulary)
        else:
            return self.original_vocab_size

    def encode(self, text: str, **kwargs):
        """
        Encode text using original tokenizer.
        Note: This doesn't apply n-gram merging - that's handled separately.

        Args:
            text: Text to encode
            **kwargs: Additional arguments for encoding

        Returns:
            Encoded result from original tokenizer
        """
        return self.original_tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text, handling both original and n-gram tokens.

        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        # Separate original tokens from n-gram tokens
        original_tokens = []

        for token_id in token_ids:
            if token_id < self.original_vocab_size:
                # Original token - can decode normally
                original_tokens.append(token_id)
            else:
                # N-gram token - convert to readable representation
                token_str = self.id_to_token(token_id)
                # For decoding, we'll represent n-grams as [token1+token2] format
                original_tokens.append(f"[{token_str}]")

        # Handle mixed case: some integers, some strings
        if any(isinstance(token, str) for token in original_tokens):
            # Mixed case - manually reconstruct
            result_parts = []
            for token in original_tokens:
                if isinstance(token, str):
                    result_parts.append(token)
                else:
                    # Decode single original token
                    decoded_token = self.original_tokenizer.decode([token], skip_special_tokens=skip_special_tokens)
                    result_parts.append(decoded_token)
            return " ".join(result_parts)
        else:
            # All original tokens - use original decoder
            return self.original_tokenizer.decode(original_tokens, skip_special_tokens=skip_special_tokens)

    def reconstruct_ngram_meaning(self, token_id: int) -> str:
        """
        Reconstruct the semantic meaning of an n-gram token.

        Args:
            token_id: N-gram token ID to reconstruct

        Returns:
            Human-readable representation of the n-gram
        """
        if token_id < self.original_vocab_size:
            return self.id_to_token(token_id)

        if self.ngram_bpe and token_id in self.ngram_bpe.id_to_pair:
            return self.ngram_bpe.reconstruct_ngram_meaning(token_id, self.extended_vocabulary)
        else:
            return self.id_to_token(token_id)

    # Delegate other methods to original tokenizer
    def __getattr__(self, name):
        """Delegate unknown methods to the original tokenizer."""
        return getattr(self.original_tokenizer, name)

    def get_original_tokenizer(self):
        """Get access to the original tokenizer if needed."""
        return self.original_tokenizer

    def get_ngram_info(self) -> Dict:
        """
        Get information about n-gram extensions.

        Returns:
            Dictionary with n-gram statistics
        """
        if self.ngram_bpe:
            return {
                'has_ngrams': True,
                'original_vocab_size': self.original_vocab_size,
                'extended_vocab_size': len(self.extended_vocabulary),
                'ngrams_created': len(self.extended_vocabulary) - self.original_vocab_size,
                'ngram_bpe_info': self.ngram_bpe.get_ngram_vocab_info()
            }
        else:
            return {
                'has_ngrams': False,
                'original_vocab_size': self.original_vocab_size,
                'extended_vocab_size': self.original_vocab_size,
                'ngrams_created': 0
            }