import pandas as pd
file_path = "../datasets/findings.csv"

data = pd.read_csv(file_path, sep=",")

print((len(data)))

exit()

import json
data_path = "/Users/emirkarayagiz/Downloads/arxiv-metadata-oai-snapshot.json"

records = []
with open(data_path, 'r') as f:
    for i, line in enumerate(f):
        if i == 300_000:  # Limit to first 1 million records for practicality
            break
        paper = json.loads(line)
        records.append({
            'abstract': paper['abstract']
        })

# Convert to DataFrame

df = pd.DataFrame(records)
df.dropna(subset=['abstract'], inplace=True)
df.to_csv("../veri_setleri/arxiv_abstract.csv", index=False)