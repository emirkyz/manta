import pandas as pd
from pathlib import Path
from collections import OrderedDict
from typing import Optional

from ..console.console_manager import ConsoleManager, get_console


def export_topics_to_excel(topics_data, output_dir, table_name, console: Optional[ConsoleManager] = None):
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if output_dir already includes the table_name to avoid double nesting
    if output_path.name == table_name:
        table_output_dir = output_path
    else:
        table_output_dir = output_path / table_name
    table_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to Excel
    excel_path = table_output_dir / f"{table_name}_topics.xlsx"
    df.to_excel(excel_path, index=False)
    _console = console or get_console()
    _console.print_debug(f"Topics exported to: {excel_path}", tag="EXPORT")

    return excel_path