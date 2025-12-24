"""Cache management for MANTA preprocessing data."""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sparse

from .processing_utils import ProcessingPaths, CachedData, ModelComponents
from ..utils.console.console_manager import ConsoleManager

# Chunk size for memory-efficient HDF5 writes
CHUNK_SIZE = 10000


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
        from previously cached files. Supports both new HDF5 format and legacy
        NPZ format for backward compatibility.

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

        # Determine which format to load (HDF5 or legacy NPZ)
        if paths.metadata_file.exists():
            # New HDF5 format
            if console:
                console.print_status(
                    f"Loading metadata from {paths.metadata_file.name} (HDF5 format)...",
                    "processing"
                )
            return CacheManager._load_from_hdf5(paths.metadata_file, tdm, console)

        elif paths.metadata_file_legacy.exists():
            # Legacy NPZ format (backward compatibility)
            if console:
                console.print_status(
                    f"Loading metadata from {paths.metadata_file_legacy.name} (legacy NPZ format)...",
                    "processing"
                )
            return CacheManager._load_from_npz(paths.metadata_file_legacy, tdm, console)

        else:
            raise FileNotFoundError(
                f"No metadata cache found at {paths.metadata_file} or {paths.metadata_file_legacy}"
            )

    @staticmethod
    def save_cached_data(
        paths: ProcessingPaths,
        data: CachedData,
        console: Optional[ConsoleManager] = None
    ) -> None:
        """Save preprocessing results to cache files using HDF5 for memory efficiency.

        Saves the TF-IDF matrix (NPZ) and metadata (HDF5) for future reuse.
        Uses chunked HDF5 writes to reduce memory usage for large datasets.

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

        # Save sparse matrix (keep as NPZ - scipy.sparse handles this efficiently)
        sparse.save_npz(paths.tfidf_matrix_file, data.tdm)

        if console:
            console.print_status("TF-IDF matrix saved.", "success")
            console.print_status(
                f"Saving metadata to {paths.metadata_file.name} (HDF5 format)...",
                "processing"
            )

        # Remove old HDF5 file if exists (fresh start)
        if paths.metadata_file.exists():
            paths.metadata_file.unlink()

        # Save metadata using HDF5 with chunked writes for memory efficiency
        CacheManager._save_to_hdf5(paths.metadata_file, data, console)

        if console:
            console.print_status("Metadata saved.", "success")

    @staticmethod
    def _serialize_datetime(datetime_series: pd.Series) -> dict:
        """Convert datetime series to serializable format.

        Stores datetime as separate year and month arrays to avoid conversion issues.
        This approach is consistent with how datetime is handled throughout the codebase.

        Args:
            datetime_series: Pandas Series with datetime values

        Returns:
            Dictionary with 'year' and 'month' NumPy arrays
        """
        # Ensure it's datetime type
        if not pd.api.types.is_datetime64_any_dtype(datetime_series):
            # Detect the type of values and convert appropriately
            # pd.to_datetime() interprets integers as nanoseconds since epoch,
            # which produces wrong dates (1970) for year values like 2000, 2020
            sample = datetime_series.dropna()
            if len(sample) > 0:
                try:
                    numeric_sample = pd.to_numeric(sample, errors='coerce').dropna()
                    if len(numeric_sample) > 0:
                        min_val = numeric_sample.min()
                        max_val = numeric_sample.max()

                        # Check if values are years (1900-2100 range)
                        if 1900 < min_val < 2100 and 1900 < max_val < 2100:
                            # Convert year integers to datetime properly
                            datetime_series = pd.to_datetime(
                                datetime_series.astype(int).astype(str),
                                format='%Y',
                                errors='coerce'
                            )
                        # Check if values are POSIX timestamps (milliseconds since epoch)
                        elif min_val > 1e12:
                            datetime_series = pd.to_datetime(datetime_series, unit='ms', errors='coerce')
                        # Check if values are POSIX timestamps (seconds since epoch)
                        elif min_val > 1e9:
                            datetime_series = pd.to_datetime(datetime_series, unit='s', errors='coerce')
                        else:
                            # Fallback to standard conversion
                            datetime_series = pd.to_datetime(datetime_series, errors='coerce')
                    else:
                        # Non-numeric values, try standard conversion
                        datetime_series = pd.to_datetime(datetime_series, errors='coerce')
                except Exception:
                    # If detection fails, try standard conversion
                    datetime_series = pd.to_datetime(datetime_series, errors='coerce')
            else:
                datetime_series = pd.to_datetime(datetime_series, errors='coerce')

        # Extract year and month components
        # Handle both Series and DatetimeIndex
        if hasattr(datetime_series, 'dt'):
            # It's a Series with datetime accessor
            return {
                'year': datetime_series.dt.year.values,
                'month': datetime_series.dt.month.values
            }
        else:
            # It's a DatetimeIndex
            return {
                'year': datetime_series.year.values,
                'month': datetime_series.month.values
            }

    @staticmethod
    def _deserialize_datetime(datetime_data) -> pd.Series:
        """Convert serialized datetime data back to pandas Series.

        Handles both new format (dict with year/month) and legacy formats
        (year values or POSIX timestamps) for backward compatibility.

        Args:
            datetime_data: Either a dict with 'year'/'month' keys or a NumPy array

        Returns:
            Pandas Series with properly typed datetime values
        """
        # New format: dictionary with year and month arrays
        if isinstance(datetime_data, dict) or (hasattr(datetime_data, 'keys') and 'year' in datetime_data):
            year = datetime_data['year']
            month = datetime_data['month']

            # Create DataFrame with year, month, and day=1
            datetime_df = pd.DataFrame({
                'year': year,
                'month': month,
                'day': 1
            })

            # Convert to datetime
            return pd.to_datetime(datetime_df)

        # Legacy format: NumPy array (backward compatibility)
        datetime_array = datetime_data
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

    # ========== HDF5 Save/Load Methods ==========

    @staticmethod
    def _save_to_hdf5(
        path: Path,
        data: CachedData,
        console: Optional[ConsoleManager] = None
    ) -> None:
        """Save metadata to HDF5 file with chunked writing for memory efficiency.

        Uses chunked writes for large arrays (especially text_array) to avoid
        loading everything into memory at once.

        Args:
            path: Path to HDF5 file
            data: CachedData object to save
            console: Optional console manager for status messages
        """
        # Variable-length string type for vocab and documents
        dt = h5py.special_dtype(vlen=str)

        with h5py.File(str(path), 'w') as f:
            # Save vocab (usually smaller, single write is fine)
            f.create_dataset('vocab', data=data.vocab, dtype=dt, compression='gzip')

            # Save documents with chunked writing (key memory optimization)
            n_docs = len(data.text_array)
            chunk_size = min(CHUNK_SIZE, n_docs) if n_docs > 0 else 1
            docs_ds = f.create_dataset(
                'documents',
                shape=(n_docs,),
                dtype=dt,
                chunks=(chunk_size,),
                compression='gzip'
            )

            # Write documents in chunks to avoid memory spike
            for start in range(0, n_docs, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, n_docs)
                docs_ds[start:end] = data.text_array[start:end]

            # Save datetime arrays if present
            if data.datetime_series is not None:
                datetime_dict = CacheManager._serialize_datetime(data.datetime_series)
                dt_grp = f.create_group('datetime')
                dt_grp.create_dataset('year', data=datetime_dict['year'], compression='gzip')
                dt_grp.create_dataset('month', data=datetime_dict['month'], compression='gzip')

            # Save pagerank weights if present
            if data.pagerank_weights is not None and len(data.pagerank_weights) > 0:
                f.create_dataset(
                    'pagerank_weights',
                    data=data.pagerank_weights,
                    compression='gzip'
                )

            # Save metadata scalars as attributes
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['datetime_is_combined'] = data.datetime_is_combined
            meta_grp.attrs['format_version'] = 2

    @staticmethod
    def _load_from_hdf5(
        path: Path,
        tdm: sparse.csr_matrix,
        console: Optional[ConsoleManager] = None
    ) -> CachedData:
        """Load metadata from HDF5 file.

        Args:
            path: Path to HDF5 file
            tdm: Pre-loaded sparse TF-IDF matrix
            console: Optional console manager for status messages

        Returns:
            CachedData object with loaded preprocessing results
        """
        with h5py.File(str(path), 'r') as f:
            # Load vocab
            vocab = [v.decode('utf-8') if isinstance(v, bytes) else v for v in f['vocab'][:]]

            # Load documents
            text_array = [d.decode('utf-8') if isinstance(d, bytes) else d for d in f['documents'][:]]

            # Load datetime if present
            datetime_series = None
            datetime_is_combined = False
            if 'datetime' in f:
                datetime_dict = {
                    'year': f['datetime/year'][:],
                    'month': f['datetime/month'][:]
                }
                datetime_series = CacheManager._deserialize_datetime(datetime_dict)
                datetime_is_combined = f['metadata'].attrs.get('datetime_is_combined', False)

            # Load pagerank weights if present
            pagerank_weights = None
            if 'pagerank_weights' in f:
                pagerank_weights = f['pagerank_weights'][:]

        if console:
            console.print_status("Loaded pre-processed data (HDF5 format).", "success")

        return CachedData(
            tdm=tdm,
            vocab=vocab,
            text_array=text_array,
            datetime_series=datetime_series,
            datetime_is_combined=datetime_is_combined,
            pagerank_weights=pagerank_weights
        )

    @staticmethod
    def _load_from_npz(
        path: Path,
        tdm: sparse.csr_matrix,
        console: Optional[ConsoleManager] = None
    ) -> CachedData:
        """Load metadata from legacy NPZ file (backward compatibility).

        Args:
            path: Path to NPZ file
            tdm: Pre-loaded sparse TF-IDF matrix
            console: Optional console manager for status messages

        Returns:
            CachedData object with loaded preprocessing results
        """
        with np.load(path, allow_pickle=True) as file:
            vocab = list(file["vocab"])
            text_array = list(file["text_array"])

            # Handle datetime series if present
            datetime_series = None
            datetime_is_combined = False

            # Try new format first (year and month arrays)
            if "datetime_year" in file and "datetime_month" in file:
                datetime_dict = {
                    'year': file["datetime_year"],
                    'month': file["datetime_month"]
                }
                datetime_series = CacheManager._deserialize_datetime(datetime_dict)
                datetime_is_combined = bool(file["datetime_is_combined"]) if "datetime_is_combined" in file else True

            # Fall back to legacy format for backward compatibility
            elif "datetime_series" in file and file["datetime_series"] is not None:
                datetime_series = CacheManager._deserialize_datetime(file["datetime_series"])
                datetime_is_combined = False

            # Load PageRank weights if present
            pagerank_weights = None
            if "pagerank_weights" in file:
                pr_weights = file["pagerank_weights"]
                if pr_weights is not None and len(pr_weights) > 0:
                    pagerank_weights = pr_weights

        if console:
            console.print_status("Loaded pre-processed data (legacy NPZ format).", "success")

        return CachedData(
            tdm=tdm,
            vocab=vocab,
            text_array=text_array,
            datetime_series=datetime_series,
            datetime_is_combined=datetime_is_combined,
            pagerank_weights=pagerank_weights
        )

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
