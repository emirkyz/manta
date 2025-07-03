import pandas as pd
from standalone_nmf import process_english_file

example_sentences = """
    I love programming
    I love programming
    I love programming
    I love programming"""
    
empty_df = pd.DataFrame({"text": [example_sentences]})
result = process_english_file(empty_df, "text", lemmatize=True)
print(result)