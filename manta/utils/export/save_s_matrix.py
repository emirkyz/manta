import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)


def _normalize_s_matrix_columns(s_matrix):
    """
    Performs L1 column-wise normalization on the S matrix.

    Each column will sum to 1.0, making the values interpretable as proportions
    or probability-like weights for topic relationships.

    Args:
        s_matrix (np.ndarray): The original S matrix (k x k).

    Returns:
        np.ndarray: Column-normalized S matrix where each column sums to 1.0.

    Note:
        - Handles zero columns by leaving them as zeros
        - Handles negative values (though NMTF should produce non-negative values)
        - Preserves the original matrix structure
    """
    s_matrix_array = np.array(s_matrix)
    normalized = s_matrix_array.copy()

    # Calculate column sums
    col_sums = np.sum(np.abs(normalized), axis=0, keepdims=True)

    # Avoid division by zero: only normalize non-zero columns
    non_zero_cols = col_sums > 0
    normalized[:, non_zero_cols.flatten()] /= col_sums[:, non_zero_cols.flatten()]

    # Log information about normalization
    n_zero_cols = np.sum(~non_zero_cols)
    if n_zero_cols > 0:
        logger.warning(f"Found {n_zero_cols} zero columns in S matrix during normalization")

    logger.info(f"Applied L1 column normalization to S matrix. "
                f"Column sums range: [{np.sum(normalized, axis=0).min():.4f}, "
                f"{np.sum(normalized, axis=0).max():.4f}]")

    return normalized


def _format_matrix_compact(matrix, indent_level=1):
    """
    Format a 2D matrix for compact JSON representation.

    Each row is formatted on a single line for better readability while
    maintaining valid JSON structure.

    Args:
        matrix (np.ndarray or list): The matrix to format (2D array).
        indent_level (int): Indentation level for the matrix (default: 1).

    Returns:
        str: Formatted matrix string with each row on a single line.

    Example:
        >>> matrix = [[0.5, 0.3], [0.1, 0.8]]
        >>> print(_format_matrix_compact(matrix, indent_level=2))
        [
            [0.5, 0.3],
            [0.1, 0.8]
          ]
    """
    indent = "  " * indent_level

    # Convert numpy array to list if needed
    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist()

    # Format each row on a single line with rounded values
    rows = []
    for row in matrix:
        # Round float values to 4 decimal places for cleaner output
        rounded_row = [round(val, 4) if isinstance(val, float) else val for val in row]
        rows.append(json.dumps(rounded_row))

    # Join rows with proper indentation
    if len(rows) == 0:
        return "[]"

    matrix_str = "[\n" + indent
    matrix_str += f",\n{indent}".join(rows)
    matrix_str += "\n" + "  " * (indent_level - 1) + "]"

    return matrix_str


def save_s_matrix(s_matrix, output_dir, table_name, data_frame_name=None):
    """
    Saves the NMTF S matrix to a JSON file with both original and normalized versions.

    In Non-negative Matrix Tri-Factorization (NMTF), the input matrix V is decomposed as:
        V ≈ W @ S @ H

    The S matrix represents the relationships between latent factors in the W and H matrices.
    This function saves both the original S matrix and a column-normalized version (L1 normalization)
    where each column sums to 1.0, making values interpretable as proportions.

    Args:
        s_matrix (np.ndarray): The S matrix from NMTF decomposition (k x k matrix).
        output_dir (str): Output directory path where the file will be saved.
        table_name (str): Name of the table/dataset for file naming.
        data_frame_name (str, optional): Alternative name for the output file.

    Returns:
        dict: Dictionary containing the S matrix data that was saved with structure:
              {
                  "metadata": {
                      "shape": [k, k],
                      "normalization": {...},
                      "created_at": "...",
                      "nmf_method": "nmtf"
                  },
                  "matrices": {
                      "original": [[...]],
                      "normalized": [[...]]
                  }
              }

    Side Effects:
        - Creates output directory if it doesn't exist
        - Saves JSON file: {table_name}_s_matrix.json
        - Logs normalization information
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
        - Includes comprehensive metadata and both matrix versions
        - Column normalization uses L1 norm (each column sums to 1.0)
    """

    # Convert numpy array to list for JSON serialization
    if isinstance(s_matrix, np.ndarray):
        s_matrix_array = s_matrix
        s_matrix_list = s_matrix.tolist()
    else:
        # If already a list or can be converted to one
        s_matrix_array = np.array(s_matrix)
        s_matrix_list = s_matrix

    # Perform L1 column normalization
    logger.info("Performing L1 column normalization on S matrix...")
    normalized_matrix = _normalize_s_matrix_columns(s_matrix_array)
    normalized_list = normalized_matrix.tolist()

    # Create output data structure with metadata and placeholder for matrices
    s_matrix_data = {
        "metadata": {
            "shape": list(s_matrix_array.shape),
            "normalization": {
                "type": "L1_column",
                "description": "Column-wise L1 normalization where each column sums to 1.0",
                "formula": "normalized[:, i] = original[:, i] / sum(abs(original[:, i]))"
            },
            "created_at": datetime.now().isoformat(),
            "nmf_method": "nmtf"
        },
        "matrices": {
            "original": "<<MATRIX_PLACEHOLDER_ORIGINAL>>",
            "normalized": "<<MATRIX_PLACEHOLDER_NORMALIZED>>"
        }
    }

    # Determine output directory
    if output_dir:
        table_output_dir = Path(output_dir)
    else:
        # Fall back to current working directory
        table_output_dir = Path.cwd() / "Output" / (data_frame_name or table_name)
        table_output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON file with compact matrix formatting
    file_name = data_frame_name or table_name
    s_matrix_file_path = table_output_dir / f"{file_name}_s_matrix.json"

    try:
        # Convert to JSON string with standard formatting
        json_str = json.dumps(s_matrix_data, indent=2, ensure_ascii=False)

        # Replace placeholders with compact-formatted matrices
        original_compact = _format_matrix_compact(s_matrix_array, indent_level=3)
        normalized_compact = _format_matrix_compact(normalized_matrix, indent_level=3)

        json_str = json_str.replace('"<<MATRIX_PLACEHOLDER_ORIGINAL>>"', original_compact)
        json_str = json_str.replace('"<<MATRIX_PLACEHOLDER_NORMALIZED>>"', normalized_compact)

        # Write the formatted JSON to file
        with open(s_matrix_file_path, "w", encoding="utf-8") as f:
            f.write(json_str)

        print(f"S matrix saved to: {s_matrix_file_path}")
    except Exception as e:
        print(f"Error saving S matrix: {e}")
        raise

    # Return data structure for compatibility (with actual lists, not placeholders)
    return {
        "metadata": s_matrix_data["metadata"],
        "matrices": {
            "original": s_matrix_list,
            "normalized": normalized_list
        }
    }


def load_s_matrix(file_path, use_normalized=True):
    """
    Loads an S matrix from a JSON file, supporting both old and new formats.

    This function provides backward compatibility with old S matrix files
    (format: {"s_matrix": [[...]], "shape": [...]}) while supporting the new format
    (with metadata and both original/normalized matrices).

    Args:
        file_path (str or Path): Path to the S matrix JSON file.
        use_normalized (bool): If True, returns the normalized version (default).
                              If False, returns the original version.
                              Only applies to new format files.

    Returns:
        np.ndarray: The S matrix as a numpy array.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid or corrupted.
        KeyError: If expected keys are missing from the JSON structure.

    Example:
        # Load normalized S matrix (default)
        s_matrix_norm = load_s_matrix("/path/to/output/analysis_s_matrix.json")

        # Load original S matrix
        s_matrix_orig = load_s_matrix("/path/to/output/analysis_s_matrix.json",
                                      use_normalized=False)

        # Works with old format files too
        s_matrix_old = load_s_matrix("/path/to/old_format_s_matrix.json")

    Note:
        - Automatically detects file format (old vs new)
        - For old format files, use_normalized parameter is ignored
        - Logs information about the loaded format
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"S matrix file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in S matrix file: {file_path}") from e

    # Check if this is the new format (has 'matrices' key)
    if "matrices" in data:
        logger.info(f"Loading S matrix from new format file: {file_path}")

        # Validate structure
        if "original" not in data["matrices"] or "normalized" not in data["matrices"]:
            raise KeyError("New format S matrix file missing 'original' or 'normalized' matrix")

        # Return requested version
        if use_normalized:
            logger.info("Using normalized S matrix (L1 column-normalized)")
            matrix_list = data["matrices"]["normalized"]
        else:
            logger.info("Using original S matrix (unnormalized)")
            matrix_list = data["matrices"]["original"]

    # Old format (backward compatibility)
    elif "s_matrix" in data:
        logger.info(f"Loading S matrix from old format file: {file_path}")
        logger.info("Old format detected - returning original unnormalized matrix")
        matrix_list = data["s_matrix"]

    else:
        raise ValueError(f"Unrecognized S matrix file format: {file_path}")

    # Convert to numpy array
    s_matrix = np.array(matrix_list)

    # Validate shape
    if len(s_matrix.shape) != 2:
        raise ValueError(f"S matrix must be 2D, got shape: {s_matrix.shape}")

    logger.info(f"Loaded S matrix with shape: {s_matrix.shape}")

    return s_matrix
