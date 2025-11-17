"""Cache management for MANTA preprocessing data."""

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from typing import Optional
from .processing_utils import ProcessingPaths, CachedData, ModelComponents
from ..utils.console.console_manager import ConsoleManager


class CacheManager:
    """Manages loading and saving of preprocessed data and model components.

    This class handles all file I/O for cached TF-IDF matrices, metadata,
    and trained model components. It provides a clean interface to avoid
    repeating complex serialization/deserialization logic.
    """

    @staticmethod
    def load_cached_data(
        paths: ProcessingPaths,
        console: Optional[ConsoleManager] = None
    ) -> CachedData:
        """Load cached preprocessing results from disk.

        Loads the TF-IDF matrix and metadata (vocab, text_array, datetime_series)
        from previously cached files.

        Args:
            paths: ProcessingPaths object with file locations
            console: Optional console manager for status messages

        Returns:
            CachedData object with loaded preprocessing results

        Raises:
            FileNotFoundError: If cache files don't exist
            ValueError: If cache files are corrupted or invalid
        """
        if console:
            console.print_status(
                f"Loading sparse matrix from {paths.tfidf_matrix_file.name}...",
                "processing"
            )

        # Load sparse TF-IDF matrix
        tdm = sparse.load_npz(paths.tfidf_matrix_file)

        if console:
            console.print_status(
                f"Loading metadata from {paths.metadata_file.name}...",
                "processing"
            )

        # Load metadata
        with np.load(paths.metadata_file, allow_pickle=True) as file:
            vocab = list(file["vocab"])
            text_array = list(file["text_array"])

            # Handle datetime series if present
            datetime_series = None
            if "datetime_series" in file and file["datetime_series"] is not None:
                datetime_series = CacheManager._deserialize_datetime(file["datetime_series"])

        if console:
            console.print_status("Loaded pre-processed data.", "success")

        return CachedData(
            tdm=tdm,
            vocab=vocab,
            text_array=text_array,
            datetime_series=datetime_series
        )

    @staticmethod
    def save_cached_data(
        paths: ProcessingPaths,
        data: CachedData,
        console: Optional[ConsoleManager] = None
    ) -> None:
        """Save preprocessing results to cache files.

        Saves the TF-IDF matrix and metadata for future reuse, allowing
        subsequent analyses to skip data loading and preprocessing.

        Args:
            paths: ProcessingPaths object with file locations
            data: CachedData object to save
            console: Optional console manager for status messages
        """
        if console:
            console.print_status(
                f"Saving sparse TF-IDF matrix to {paths.tfidf_matrix_file.name}...",
                "processing"
            )

        # Save sparse matrix
        sparse.save_npz(paths.tfidf_matrix_file, data.tdm)

        if console:
            console.print_status("TF-IDF matrix saved.", "success")
            console.print_status(
                f"Saving metadata to {paths.metadata_file.name}...",
                "processing"
            )

        # Serialize datetime if present
        datetime_array = None
        if data.datetime_series is not None:
            datetime_array = CacheManager._serialize_datetime(data.datetime_series)

        # Save metadata
        np.savez_compressed(
            paths.metadata_file,
            vocab=data.vocab,
            text_array=data.text_array,
            datetime_series=datetime_array
        )

        if console:
            console.print_status("Metadata saved.", "success")

    @staticmethod
    def _serialize_datetime(datetime_series: pd.Series) -> np.ndarray:
        """Convert datetime series to serializable format.

        Converts datetime values to POSIX timestamps (integers) for storage.

        Args:
            datetime_series: Pandas Series with datetime values

        Returns:
            NumPy array of integer timestamps
        """
        if pd.api.types.is_datetime64_any_dtype(datetime_series):
            # Convert to POSIX timestamp (seconds since epoch)
            return datetime_series.astype('int64') // 10**9
        else:
            # Already numeric, return as-is
            return datetime_series.values

    @staticmethod
    def _deserialize_datetime(datetime_array: np.ndarray) -> pd.Series:
        """Convert serialized datetime array back to pandas Series.

        Handles both year values (1900-2100) and POSIX timestamps.

        Args:
            datetime_array: NumPy array of datetime values

        Returns:
            Pandas Series with properly typed datetime values
        """
        datetime_series = pd.Series(datetime_array)

        # Check if values are years (between 1900-2100)
        if datetime_series.min() > 1900 and datetime_series.max() < 2100:
            # These are year values - convert to datetime
            year_strings = datetime_series.astype(int).astype(str)
            return pd.to_datetime(year_strings, format='%Y')

        # Check if values are POSIX timestamps (> 1 billion = after 2001)
        elif datetime_series.min() > 1e9:
            # These are POSIX timestamps - convert with unit='s'
            return pd.to_datetime(datetime_series, unit='s')

        # Return as-is if neither pattern matches
        return datetime_series

    @staticmethod
    def save_model_components(
        paths: ProcessingPaths,
        components: ModelComponents,
        variant_table_name: Optional[str] = None,
        console: Optional[ConsoleManager] = None
    ) -> None:
        """Save trained model components to disk.

        Args:
            paths: ProcessingPaths object with file locations
            components: ModelComponents object with W, H, vocab, etc.
            variant_table_name: Optional variant name for NMF-specific models
            console: Optional console manager for status messages

        Raises:
            IOError: If unable to write to disk
        """
        if console:
            console.print_status("Saving model components...", "processing")

        # Get the appropriate model file path
        model_file = paths.model_file(variant_table_name)

        # Ensure the output directory exists
        model_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save all components
            np.savez_compressed(model_file, **components.to_dict())

            if console:
                console.print_status(
                    f"Model components saved to {model_file.name}",
                    "success"
                )

        except Exception as e:
            if console:
                console.print_status(
                    f"Warning: Failed to save model components: {e}",
                    "warning"
                )
            raise

    @staticmethod
    def load_model_components(
        paths: ProcessingPaths,
        variant_table_name: Optional[str] = None,
        console: Optional[ConsoleManager] = None
    ) -> ModelComponents:
        """Load trained model components from disk.

        Args:
            paths: ProcessingPaths object with file locations
            variant_table_name: Optional variant name for NMF-specific models
            console: Optional console manager for status messages

        Returns:
            ModelComponents object with loaded W, H, vocab, etc.

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_file = paths.model_file(variant_table_name)

        if console:
            console.print_status(
                f"Loading model components from {model_file.name}...",
                "processing"
            )

        with np.load(model_file, allow_pickle=True) as data:
            components = ModelComponents(
                W=data['W'],
                H=data['H'],
                vocab=list(data['vocab']),
                text_array=list(data['text_array']),
                S=data['S'] if 'S' in data else None
            )

        if console:
            console.print_status("Model components loaded.", "success")

        return components
