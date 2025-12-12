"""Processing utilities and dataclasses for MANTA topic analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import scipy.sparse as sparse


@dataclass
class ProcessingPaths:
    """Manages file paths for caching and model output.

    Centralizes all file path generation to ensure consistent naming
    and easy path management throughout the processing pipeline.

    Attributes:
        output_dir: Base output directory for all files
        table_name: Unique identifier for this analysis run (includes topic count)
        preprocessing_name: Identifier for preprocessing cache (excludes topic count)
    """
    output_dir: Path
    table_name: str
    preprocessing_name: str

    @property
    def tfidf_matrix_file(self) -> Path:
        """Path to cached TF-IDF matrix file.

        Uses preprocessing_name (without topic count) since TFIDF is
        independent of the number of topics.
        """
        return self.output_dir / f"{self.preprocessing_name}_tfidf_matrix.npz"

    @property
    def metadata_file(self) -> Path:
        """Path to cached metadata file (vocab, text_array, datetime).

        Uses preprocessing_name (without topic count) since preprocessing
        is independent of the number of topics.
        """
        return self.output_dir / f"{self.preprocessing_name}_tfidf_metadata.npz"

    def table_output_dir(self, variant_table_name: Optional[str] = None) -> Path:
        """Get output directory for a specific table/variant.

        Args:
            variant_table_name: Optional variant name (e.g., for different NMF types)
                              If None, uses the base table_name

        Returns:
            Path to the variant-specific output directory
        """
        name = variant_table_name if variant_table_name else self.table_name
        return self.output_dir / name

    def model_file(self, variant_table_name: Optional[str] = None) -> Path:
        """Path to saved model components file.

        Args:
            variant_table_name: Optional variant name for NMF-specific models

        Returns:
            Path to model components NPZ file
        """
        name = variant_table_name if variant_table_name else self.table_name
        table_dir = self.table_output_dir(variant_table_name)
        return table_dir / f"{name}_model_components.npz"

    def cache_exists(self) -> bool:
        """Check if cached preprocessing files exist.

        Returns:
            True if both TF-IDF matrix and metadata files exist
        """
        return self.tfidf_matrix_file.exists() and self.metadata_file.exists()


@dataclass
class CachedData:
    """Container for cached preprocessing results.

    Holds all the data required to skip data loading and preprocessing stages.
    This allows for faster iteration when running multiple analyses on the same dataset.

    Attributes:
        tdm: Term-document matrix (sparse format)
        vocab: Vocabulary list (words/tokens)
        text_array: Original text documents
        datetime_series: Optional datetime values for temporal analysis
        datetime_is_combined: Whether datetime was created from combined year/month columns
        pagerank_weights: Optional PageRank weights for TF-IDF boosting (range [1, 2])
    """
    tdm: sparse.csr_matrix
    vocab: List[str]
    text_array: List[str]
    datetime_series: Optional[pd.Series] = None
    datetime_is_combined: bool = False
    pagerank_weights: Optional[np.ndarray] = None

    def __len__(self) -> int:
        """Return number of documents in the cached data."""
        return len(self.text_array)

    @property
    def has_datetime(self) -> bool:
        """Check if datetime information is available."""
        return self.datetime_series is not None and len(self.datetime_series) > 0

    @property
    def has_pagerank_weights(self) -> bool:
        """Check if PageRank weights are available."""
        return self.pagerank_weights is not None and len(self.pagerank_weights) > 0


@dataclass
class ModelComponents:
    """Container for NMF model output components.

    Structured representation of NMF decomposition results and metadata.

    Attributes:
        W: Document-topic matrix
        H: Topic-word matrix
        vocab: Vocabulary list
        text_array: Original documents
        S: Optional coupling matrix (for NMTF only)
    """
    W: any  # numpy array or sparse matrix
    H: any  # numpy array or sparse matrix
    vocab: List[str]
    text_array: List[str]
    S: Optional[any] = None  # NMTF coupling matrix

    def to_dict(self) -> dict:
        """Convert to dictionary format for saving.

        Returns:
            Dictionary with all non-None components
        """
        result = {
            'W': self.W,
            'H': self.H,
            'vocab': self.vocab,
            'text_array': self.text_array,
        }
        if self.S is not None:
            result['S'] = self.S
        return result

    @classmethod
    def from_nmf_output(cls, nmf_output: dict, vocab: List[str],
                       text_array: List[str]) -> 'ModelComponents':
        """Create ModelComponents from NMF pipeline output.

        Args:
            nmf_output: Dictionary from ModelingPipeline with W, H, and optional S
            vocab: Vocabulary list
            text_array: Original text documents

        Returns:
            ModelComponents instance
        """
        return cls(
            W=nmf_output["W"],
            H=nmf_output["H"],
            vocab=vocab,
            text_array=text_array,
            S=nmf_output.get("S")  # None if not present (NMF vs NMTF)
        )
