import json
import numpy as np
from pathlib import Path


def save_s_matrix(s_matrix, output_dir, table_name, data_frame_name=None):
    """
    Saves the NMTF S matrix to a JSON file.

    In Non-negative Matrix Tri-Factorization (NMTF), the input matrix V is decomposed as:
        V ≈ W @ S @ H

    The S matrix represents the relationships between latent factors in the W and H matrices.
    This function saves the S matrix to a JSON file for later analysis and visualization.

    Args:
        s_matrix (np.ndarray): The S matrix from NMTF decomposition (k x k matrix).
        output_dir (str): Output directory path where the file will be saved.
        table_name (str): Name of the table/dataset for file naming.
        data_frame_name (str, optional): Alternative name for the output file.

    Returns:
        dict: Dictionary containing the S matrix data that was saved.
              Format: {"s_matrix": [[...]], "shape": [k, k]}

    Side Effects:
        - Creates output directory if it doesn't exist
        - Saves JSON file: {table_name}_s_matrix.json
        - Prints confirmation message with file path

    Example:
        s_matrix = np.array([[0.5, 0.3], [0.2, 0.8]])
        result = save_s_matrix(
            s_matrix=s_matrix,
            output_dir="/path/to/output",
            table_name="my_analysis",
            data_frame_name="my_analysis"
        )
        # Creates: /path/to/output/my_analysis_s_matrix.json

    Note:
        - Converts numpy array to list for JSON serialization
        - Uses ensure_ascii=False to support Unicode characters
        - Includes matrix shape metadata for validation
    """

    # Convert numpy array to list for JSON serialization
    if isinstance(s_matrix, np.ndarray):
        s_matrix_list = s_matrix.tolist()
    else:
        # If already a list or can be converted to one
        s_matrix_list = s_matrix

    # Create output data structure
    s_matrix_data = {
        "s_matrix": s_matrix_list,
        "shape": list(np.array(s_matrix).shape)
    }

    # Determine output directory
    if output_dir:
        table_output_dir = Path(output_dir)
    else:
        # Fall back to current working directory
        table_output_dir = Path.cwd() / "Output" / (data_frame_name or table_name)
        table_output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    file_name = data_frame_name or table_name
    s_matrix_file_path = table_output_dir / f"{file_name}_s_matrix.json"

    try:
        with open(s_matrix_file_path, "w", encoding="utf-8") as f:
            json.dump(s_matrix_data, f, indent=4, ensure_ascii=False)
        print(f"S matrix saved to: {s_matrix_file_path}")
    except Exception as e:
        print(f"Error saving S matrix: {e}")
        raise

    return s_matrix_data
