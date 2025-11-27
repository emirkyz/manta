import pandas as pd

def sliding_window(sequence, window_size, step=1):
    """Generate a list of substrings (windows) of given length moving by step."""
    return [sequence[i:i+window_size] for i in range(0, len(sequence) - window_size + 1, step)]

# Parameters for sliding window
WINDOW_SIZE = 6
STEP_SIZE = 1  # slide by 1 character

# List of files and sources
data_sources = [
    ("chimpanzee.txt", "chimpanzee"),
    ("dog.txt", "dog"),
    ("human.txt", "human"),
]

dataframes = []

for filename, species in data_sources:
    data = pd.read_table(filename)
    print(data.head())

    # Apply sliding window mechanism
    data['DNA_sequence'] = data['sequence'].apply(lambda x: sliding_window(x, WINDOW_SIZE, STEP_SIZE))
    print(data[['sequence','DNA_sequence']].head())
    
    # Join sliding windows with space
    data['DNA_sequence'] = data['DNA_sequence'].apply(lambda x: ' '.join(x))
    print(data[['DNA_sequence']].head())
    
    data['source'] = species
    dataframes.append(data[['DNA_sequence', 'source']])

# Combine all sources
combined_data = pd.concat(dataframes, ignore_index=True)
print(combined_data.head())

combined_data.to_csv('combined_dna_sequence.csv', index=False)