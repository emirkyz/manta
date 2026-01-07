import json
from pathlib import Path
from typing import Optional

from ..console.console_manager import ConsoleManager, get_console


def save_doc_score_pair(doc_result, base_dir, output_dir, table_name, data_frame_name=None, console: Optional[ConsoleManager] = None):
    """
    Processes and saves topic analysis results to files and database.
    
    This function handles saving both document scores and topic word scores from topic modeling analysis.
    It can save results to JSON files in a directory structure and optionally to a database.
    
    Args:
        doc_result (dict): Document analysis results mapping topic IDs to document-score pairs.
        base_dir (str): Base directory path for output files.
        output_dir (str): Optional specific output directory path.
        table_name (str): Name of the table/dataset for file naming.
        topics_data (dict): Topic word scores mapping topic names to word-score pairs.
        result: Additional result data (unused).
        data_frame_name (str, optional): Name for database table and file paths.
        topics_db_eng (sqlalchemy.engine, optional): Database engine for saving to DB.
    
    Returns:
        dict: The document analysis results that were saved.
              
    Side Effects:
        - Creates output directory structure if it doesn't exist
        - Saves JSON file with document scores: top_docs_{data_frame_name}.json
        - Saves topics to database if engine is provided
        - Prints warning if no database engine is provided
        
    Example:
        doc_result = {
            "Topic 0": {"0": "doc1 text:0.85", "1": "doc2 text:0.75"},
            "Topic 1": {"0": "doc3 text:0.90", "1": "doc4 text:0.80"}
        }
        
        result = save_doc_score_pair(
            doc_result=doc_result,
            base_dir="/path/to/base",
            output_dir=None, 
            table_name="my_table",
            topics_data=None,
            result=None,
            data_frame_name="my_analysis",
            topics_db_eng=db_engine
        )
    
    Note:
        - Will create output directory if it doesn't exist
        - Database saving is optional
        - JSON files use UTF-8 encoding
    """
    
    _console = console or get_console()
    # Convert the topics_data format to the desired format
    if data_frame_name:
        if output_dir:  # output_dir is provided
            table_output_dir = Path(output_dir)
        else:
            # create output dir in the current working directory
            table_output_dir = Path.cwd() / "Output" / data_frame_name
            table_output_dir.mkdir(parents=True, exist_ok=True)

        # Save document scores to table-specific subdirectory
        document_file_path = table_output_dir / f"{data_frame_name}_top_docs.json"
        with open(document_file_path, "w", encoding="utf-8") as f:
            json.dump(doc_result, f, ensure_ascii=False)
        _console.print_debug(f"Document scores saved to {document_file_path}", tag="EXPORT")
    # Save topics to database

    return doc_result