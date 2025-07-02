import pandas as pd
import os
from collections import OrderedDict

def export_topics_to_excel(topics_data, output_dir, table_name):
    """
    Export topics data to Excel with words sorted by topics:
    - First column contains words ordered by their topic appearance
    - Each topic has its own column
    - Cells contain scores only if the word appears in that topic
    """
    # Create an ordered dictionary to maintain word order by topic
    ordered_words = OrderedDict()
    
    # Collect words in order of topics
    for topic_name, topic_words in topics_data.items():
        # Sort words within each topic by their score (descending)
        sorted_topic_words = sorted(topic_words.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_topic_words:
            if word not in ordered_words:
                ordered_words[word] = {}
    
    # Create DataFrame structure
    df_data = {'Word': list(ordered_words.keys())}
    
    # Add topic columns
    for topic_name, topic_words in topics_data.items():
        df_data[topic_name] = [topic_words.get(word, '') for word in df_data['Word']]
    
    # Create DataFrame and export to Excel
    df = pd.DataFrame(df_data)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if output_dir already includes the table_name to avoid double nesting
    if output_dir.endswith(table_name):
        table_output_dir = output_dir
    else:
        table_output_dir = os.path.join(output_dir, table_name)
    os.makedirs(table_output_dir, exist_ok=True)
    
    # Save to Excel
    excel_path = os.path.join(table_output_dir, f"{table_name}_topics.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Topics exported to: {excel_path}")
    
    return excel_path