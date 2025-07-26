#!/usr/bin/env python3
"""
Quick test script to verify English tokenizer implementation
"""
import pandas as pd
from manta._functions.english.english_entry import process_english_file

# Create test data
test_data = {
    'text': [
        "This is a test sentence.",
        "Another test sentence for English processing.",
        "We are testing the new tokenizer implementation.",
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing with tokenizers is powerful."
    ]
}

df = pd.DataFrame(test_data)

print("Testing English tokenizer implementation...")
print(f"Input data: {len(df)} documents")

try:
    # Test with WordPiece tokenizer
    result = process_english_file(
        df=df,
        desired_columns='text',
        lemmatize=False,
        tokenizer=None,
        tokenizer_type='wordpiece',
        emoji_map=None
    )
    
    tdm, sozluk, sayisal_veri, tokenizer, metin_array, emoji_map = result
    
    print("✅ English tokenizer implementation successful!")
    print(f"TF-IDF matrix shape: {tdm.shape}")
    print(f"Vocabulary size: {len(sozluk)}")
    print(f"Number of processed documents: {len(metin_array)}")
    print(f"Tokenizer type: {type(tokenizer)}")

    print(f"Sample vocabulary: {sozluk[:10]}")  # Print first 10 tokens
    print(f"Sample processed text: {metin_array[:3]}")  # Print first 3 processed texts
except Exception as e:
    print(f"❌ Error testing English tokenizer: {e}")
    import traceback
    traceback.print_exc()