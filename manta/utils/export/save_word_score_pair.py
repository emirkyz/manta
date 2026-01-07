import os
import json
from typing import Optional

from ..console.console_manager import ConsoleManager, get_console


def save_word_score_pair(base_dir, output_dir, table_name, topics_data, result, data_frame_name=None, topics_db_eng=None, console: Optional[ConsoleManager] = None):
    """
    Processes topic word scores and saves them as a JSON file in a structured directory format.
    
    This function takes topic data containing word-score pairs in string format, parses them
    into a structured dictionary, and saves the result to a JSON file. The output is organized
    in a directory structure under the specified base directory.
    
    Args:
        base_dir (str): The base directory path where the output folder will be created.
        output_dir (str): Output directory parameter (currently unused in implementation).
        table_name (str): Name of the table/dataset, used for creating subdirectories and naming the output file.
        topics_data (dict): Dictionary where keys are topic names and values are lists of 
                           word-score strings in format "word:score" or "multi word:score".
        result: Additional result parameter (currently unused in implementation).
    
    Returns:
        dict: A dictionary where keys are topic names and values are dictionaries mapping
              words to their corresponding float scores.
              
    Side Effects:
        - Creates directory structure: {base_dir}/Output/{table_name}/
        - Saves JSON file: {table_name}_wordcloud_scores.json in the created directory
        - Prints status messages for successful saves or errors
        
    Example:
        topics_data = {
            "topic1": ["machine learning:0.85", "data science:0.75"],
            "topic2": ["neural network:0.90", "deep learning:0.88"]
        }
        
        result = save_doc_score_pair("/path/to/base", "", "my_table", topics_data, None)
        # Creates: /path/to/base/Output/my_table/my_table_wordcloud_scores.json
        # Returns: {"topic1": {"machine learning": 0.85, "data science": 0.75}, ...}
    
    Note:
        - Words containing colons will be handled correctly by joining all parts except the last
        - Invalid word-score entries are skipped with error messages printed to console
        - The function ensures ASCII compatibility is maintained in JSON output
    """
    
    _console = console or get_console()
    # Convert the topics_data format to the desired format
    topic_word_scores = {}
    for topic_name, word_scores in topics_data.items():
        topic_dict = {}
        for word_score in word_scores:
            if word_score:  # Check if not None
                try:
                    splits = word_score.split(":")
                    word = splits[:-1]
                    score = splits[-1]
                    word = " ".join(word)  # Join back the word parts
                    topic_dict[word] = float(score)
                except:
                    _console.print_error(f"Error parsing word score: {word_score}", tag="EXPORT")
        topic_word_scores[topic_name] = topic_dict

    # Save the topic word scores to a JSON file
    if output_dir:
        # Use the provided output_dir
        table_output_dir = output_dir
        os.makedirs(table_output_dir, exist_ok=True)
    else:
        # Fall back to original behavior when output_dir is not provided
        base_dir = os.path.abspath(base_dir)
        output_dir_fallback = os.path.join(base_dir, "Output")
        os.makedirs(output_dir_fallback, exist_ok=True)

        # Create table-specific subdirectory under output folder
        table_output_dir = os.path.join(output_dir_fallback, table_name)
        os.makedirs(table_output_dir, exist_ok=True)

    # Save to table-specific subdirectory
    wordcloud_file = os.path.join(table_output_dir, f"{table_name}_word_scores.json")
    try:
        with open(wordcloud_file, "w") as f:
            json.dump(topic_word_scores, f, indent=4, ensure_ascii=False)
        _console.print_debug(f"Topic word scores saved to: {wordcloud_file}", tag="EXPORT")
    except Exception as e:
        _console.print_error(f"Error saving topic word scores: {e}", tag="EXPORT")
    return topic_word_scores