import json
import pandas as pd
from pathlib import Path


def convert_json_to_excel(word_json_data, doc_json_data, output_dir, data_frame_name,total_docs_count=None):
    """
    Converts JSON wordcloud and document data to Excel with multiple sheets:
    - Sheet 1: Summary/Title page (empty or with basic info)
    - Sheet 2: Word scores for each topic
    - Sheet 3: Document per topic table
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    excel_file = output_dir / f"{data_frame_name}_topic_analysis.xlsx"
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Sheet 1: Summary/Title page (empty or with basic info)
        summary_data = pd.DataFrame({
            'Analysis Summary': [
                f'Topic Analysis Results for: {data_frame_name}',
                f'Total Topics: {len(word_json_data)}',
                f'Total Documents: {total_docs_count}' if total_docs_count is not None else 'Total Documents: Not Provided',
                '',
                'Sheet 2: Word Scores by Topic',
                'Sheet 3: Documents per Topic'
            ]
        })
        summary_data.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Word table with topics as columns
        topics = sorted(word_json_data.keys())
        max_words = max(len(words) for words in word_json_data.values()) if word_json_data else 0
        
        word_table_rows = []
        for rank in range(1, max_words + 1):
            row = {'Rank': rank}
            for topic in topics:
                words = word_json_data[topic]
                sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
                
                if rank <= len(sorted_words):
                    word, score = sorted_words[rank - 1]
                    row[topic] = word
                else:
                    row[topic] = ""
            
            word_table_rows.append(row)
        
        df_word_table = pd.DataFrame(word_table_rows)
        df_word_table.to_excel(writer, sheet_name='Word Scores', index=False)
        
        # Sheet 3: Document table with each document in its own row
        doc_table_rows = []
        for topic in sorted(doc_json_data.keys()):
            docs = doc_json_data[topic]
            # Sort documents by score (descending)
            sorted_docs = []
            for doc_id, doc_info in docs.items():
                if ':' in doc_info:
                    doc_text, score = doc_info.rsplit(':', 1)
                    sorted_docs.append((doc_text, float(score)))
            
            sorted_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Add each document as a separate row
            for doc_text, score in sorted_docs:
                doc_table_rows.append({
                    'Topic': topic,
                    'Document': doc_text
                })
        
        df_docs = pd.DataFrame(doc_table_rows)
        df_docs.to_excel(writer, sheet_name='Documents per Topic', index=False)
    
    print(f"Excel file saved to: {excel_file}")
    return excel_file


# Usage
if __name__ == "__main__":
    # Example usage
    word_data = {"Topic 1": {"word1": 0.8, "word2": 0.6}, "Topic 2": {"word3": 0.9, "word4": 0.7}}
    doc_data = {"Topic 1": {"0": "Document text 1:0.85"}, "Topic 2": {"1": "Document text 2:0.90"}}
    
    excel_file = convert_json_to_excel(word_data, doc_data, Path.cwd(), "example_analysis")
    print(f"Excel file created: {excel_file}")