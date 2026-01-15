"""
Data loading and preprocessing pipeline for MANTA topic analysis.
"""

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

from ..utils.console.console_manager import ConsoleManager, get_console
from ..utils.database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class DataPipeline:
    """Handles data loading and preprocessing operations."""

    @staticmethod
    def validate_inputs(
        filepath: Optional[str],
        desired_columns: str,
        options: Dict[str, Any],
        dataframe: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Validate input parameters for processing.

        Args:
            filepath: Path to input file (optional if dataframe provided)
            desired_columns: Column name containing text data
            options: Configuration options
            dataframe: Optional DataFrame input (alternative to filepath)

        Raises:
            ValueError: If inputs are invalid
            FileNotFoundError: If file doesn't exist
        """
        # Ensure either filepath or dataframe is provided, but not both
        if filepath is None and dataframe is None:
            raise ValueError("Either filepath or dataframe must be provided")

        if filepath is not None and dataframe is not None:
            raise ValueError("Cannot provide both filepath and dataframe - choose one")

        # Validate filepath if provided
        if filepath is not None and not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")

        # Validate dataframe if provided
        if dataframe is not None:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("dataframe must be a pandas DataFrame")
            if dataframe.empty:
                raise ValueError("dataframe cannot be empty")

        if not desired_columns or not desired_columns.strip():
            raise ValueError("desired_columns cannot be empty")

        required_options = ["LANGUAGE", "DESIRED_TOPIC_COUNT", "N_TOPICS"]
        for option in required_options:
            if option not in options:
                raise ValueError(f"Missing required option: {option}")

        if options["LANGUAGE"] not in ["TR", "EN"]:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}. Must be 'TR' or 'EN'")

    @staticmethod
    def load_data(
        filepath: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        options: Optional[Dict[str, Any]] = None,
        console: Optional[ConsoleManager] = None,
    ) -> pd.DataFrame:
        """
        Load data from either a file or a DataFrame.

        Args:
            filepath: Path to input file (CSV or Excel)
            dataframe: Pre-loaded DataFrame
            options: Configuration options containing separator and filter settings
            console: Console manager for status messages

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If neither or both filepath and dataframe are provided
        """
        if filepath is None and dataframe is None:
            raise ValueError("Either filepath or dataframe must be provided")

        if filepath is not None and dataframe is not None:
            raise ValueError("Cannot provide both filepath and dataframe")

        # Load from file if filepath provided
        if filepath is not None:
            return DataPipeline.load_data_file(filepath, options, console)

        # Use provided dataframe
        if console:
            console.print_status(f"Using provided DataFrame with {len(dataframe)} rows", "info")

        # Apply data filters if specified
        df = DataPipeline._apply_data_filters(dataframe, options, console)
        return df

    @staticmethod
    def load_data_file(
        filepath: str, options: Dict[str, Any], console: Optional[ConsoleManager] = None
    ) -> pd.DataFrame:
        """
        Load data from CSV or Excel file.

        Args:
            filepath: Path to input file
            options: Configuration options containing separator and filter settings
            console: Console manager for status messages

        Returns:
            Loaded DataFrame
        """
        _console = console or get_console()
        _console.print_status("Reading input file...", "processing")

        if str(filepath).endswith(".csv"):
            # Read the CSV file with the specified separator
            df = pd.read_csv(
                filepath,
                encoding="utf-8",
                sep=options["separator"],
                engine="python",
                on_bad_lines="skip",
                # nrows=1_000, #TODO: will be removed when we have a better way to handle large files
            )
            # df = df.sample(n = 1_000)

            # Track initial data load
            initial_row_count = len(df)
            _console.print_debug(f"Initial rows from CSV: {initial_row_count}", tag="DATA LOADING")

            # Track year filter
            if "year" in df.columns:
                before_year_filter = len(df)
                df = df[df["year"] < 2026]
                after_year_filter = len(df)
                rows_removed = before_year_filter - after_year_filter
                if rows_removed > 0:
                    _console.print_debug(f"Removed {rows_removed} rows with year >= 2026", tag="YEAR FILTER")
                    _console.print_debug(f"  Before: {before_year_filter} rows, After: {after_year_filter} rows", tag="YEAR FILTER")
            else:
                pass

        elif str(filepath).endswith(".xlsx") or str(filepath).endswith(".xls"):
            df = pd.read_excel(filepath)

        # Apply data filters if specified
        df = DataPipeline._apply_data_filters(df, options, _console)
        return df

    @staticmethod
    def _apply_data_filters(
        df: pd.DataFrame, options: Dict[str, Any], console: Optional[ConsoleManager] = None
    ) -> pd.DataFrame:
        """Apply data filters based on configuration options."""
        _console = console or get_console()
        try:
            if options.get("filter_app", False):
                initial_filter_count = len(df)
                _console.print_debug(f"Starting with {initial_filter_count} rows", tag="DATA FILTERS")

                filter_options = options.get("data_filter_options", {})
                if filter_options.get("filter_app_country", ""):
                    country_col = filter_options.get("filter_app_country_column", "")
                    if country_col in df.columns:
                        before_country = len(df)
                        df = df[df[country_col].str.upper() == filter_options["filter_app_country"].upper()]
                        after_country = len(df)
                        removed = before_country - after_country
                        _console.print_debug(f"Removed {removed} rows", tag="COUNTRY FILTER")
                        _console.print_debug(f"  Filtered for country: {filter_options['filter_app_country']}", tag="COUNTRY FILTER")
                        _console.print_debug(f"  Before: {before_country}, After: {after_country}", tag="COUNTRY FILTER")
                        _console.print_status(
                            f"Applied country filter: {filter_options['filter_app_country']}",
                            "info",
                        )
                    else:
                        _console.print_warning(f"Filter column '{country_col}' not found in data")

                if filter_options.get("filter_app_name", ""):
                    app_col = filter_options.get("filter_app_column", "")
                    if app_col in df.columns:
                        before_app = len(df)
                        df = df[df[app_col] == filter_options["filter_app_name"]]
                        after_app = len(df)
                        removed = before_app - after_app
                        _console.print_debug(f"Removed {removed} rows", tag="APP FILTER")
                        _console.print_debug(f"  Filtered for app: {filter_options['filter_app_name']}", tag="APP FILTER")
                        _console.print_debug(f"  Before: {before_app}, After: {after_app}", tag="APP FILTER")
                        _console.print_status(
                            f"Applied app filter: {filter_options['filter_app_name']}", "info"
                        )
                    else:
                        _console.print_warning(f"Filter column '{app_col}' not found in data")

                # Summary of filtering
                total_filtered = initial_filter_count - len(df)
                if total_filtered > 0:
                    _console.print_debug(f"Total rows removed by filters: {total_filtered}", tag="FILTERS SUMMARY")
        except KeyError as e:
            _console.print_warning(f"Missing filter configuration: {e}")
        except Exception as e:
            _console.print_warning(f"Error applying data filters: {e}")

        return df

    @staticmethod
    def preprocess_dataframe(
        df: pd.DataFrame,
        desired_columns: str,
        options: Dict[str, Any],
        main_db_eng,
        table_name: str,
        console: Optional[ConsoleManager] = None,
    ) -> pd.DataFrame:
        """
        Preprocess the loaded DataFrame.

        Args:
            df: Raw DataFrame
            desired_columns: Column containing text data
            options: Configuration options
            main_db_eng: Database engine for main data
            table_name: Name for database table
            console: Console manager for status messages

        Returns:
            Preprocessed DataFrame
        """
        _console = console or get_console()
        _console.print_status("Preprocessing data...", "processing")

        # Select only desired columns and validate they exist
        if desired_columns not in df.columns:
            available_columns = ", ".join(df.columns.tolist())
            raise KeyError(
                f"Column '{desired_columns}' not found in data. Available columns: {available_columns}"
            )

        # Check for PageRank column to use for TF-IDF weighting
        pagerank_col = options.get("pagerank_column")
        has_pagerank = pagerank_col and pagerank_col in df.columns
        if pagerank_col and pagerank_col not in df.columns:
            _console.print_warning(
                f"PageRank column '{pagerank_col}' not found in data. Skipping PageRank weighting.",
                tag="PAGERANK"
            )
            options["pagerank_column"] = None
            has_pagerank = False

        # Check for datetime columns to preserve for temporal analysis
        common_datetime_cols = [
            "year",
            "date",
            "datetime",
            "timestamp",
            "time",
            "rev_submit_millis_since_epoch",
            "created_at",
            "updated_at",
        ]
        datetime_col_found = None

        # Check if both 'year' and 'month' columns exist - combine them into MM-YYYY format
        if "year" in df.columns and "month" in df.columns:
            _console.print_status(
                "Combining 'year' and 'month' columns into datetime...", "processing"
            )

            # Make a copy to avoid SettingWithCopyWarning
            df = df.copy()

            # Create a helper function to convert month to numeric
            def convert_month_to_numeric(month_val):
                """Convert month name or number to numeric (1-12)."""
                if pd.isna(month_val):
                    return 1  # Default to January if missing

                # If already numeric, validate and return
                if isinstance(month_val, (int, float)):
                    month_num = int(month_val)
                    return month_num if 1 <= month_num <= 12 else 1

                # Convert string month names to numbers
                month_str = str(month_val).strip()

                # Check for None values
                if month_str is None or month_str == "None":
                    logger.warning(
                        f"Invalid month value encountered: {month_val}. Defaulting to January."
                    )
                    return 1

                month_map = {
                    "jan": 1,
                    "january": 1,
                    "feb": 2,
                    "february": 2,
                    "mar": 3,
                    "march": 3,
                    "apr": 4,
                    "april": 4,
                    "may": 5,
                    "jun": 6,
                    "june": 6,
                    "jul": 7,
                    "july": 7,
                    "aug": 8,
                    "august": 8,
                    "sep": 9,
                    "sept": 9,
                    "september": 9,
                    "oct": 10,
                    "october": 10,
                    "nov": 11,
                    "november": 11,
                    "dec": 12,
                    "december": 12,
                }

                # Try to parse as month name
                month_lower = month_str.lower()
                if month_lower in month_map:
                    return month_map[month_lower]

                # Try to parse as number
                try:
                    month_num = int(month_str)
                    return month_num if 1 <= month_num <= 12 else 1
                except ValueError:
                    return 1  # Default to January if can't parse

            # Apply conversion to month column
            df["month_numeric"] = df["month"].apply(convert_month_to_numeric)

            # Combine year and month into a datetime column (day=1 for proper sorting)
            df["datetime_combined"] = pd.to_datetime(
                df["year"].astype(int).astype(str)
                + "-"
                + df["month_numeric"].astype(int).astype(str).str.zfill(2)
                + "-01",
                format="%Y-%m-%d",
                errors="coerce",
            )

            # Store that we combined year and month
            datetime_col_found = "datetime_combined"
            options["datetime_column"] = datetime_col_found
            options["datetime_is_combined_year_month"] = True

            # Select columns to keep (include pagerank if available)
            cols_to_keep = [desired_columns, "datetime_combined"]
            if has_pagerank:
                cols_to_keep.append(pagerank_col)
            df = df[cols_to_keep]

            _console.print_status(
                f"Created combined datetime column from year and month", "info"
            )
        else:
            # Original logic for other datetime columns
            for col in common_datetime_cols:
                if col in df.columns:
                    datetime_col_found = col
                    break

            # Select text column and datetime column if available (include pagerank if available)
            if datetime_col_found:
                cols_to_keep = [desired_columns, datetime_col_found]
                if has_pagerank:
                    cols_to_keep.append(pagerank_col)
                df = df[cols_to_keep]
                options["datetime_column"] = datetime_col_found  # Store for later use in pipeline
                options["datetime_is_combined_year_month"] = False
                _console.print_status(f"Preserved datetime column: {datetime_col_found}", "info")
            else:
                cols_to_keep = [desired_columns]
                if has_pagerank:
                    cols_to_keep.append(pagerank_col)
                df = df[cols_to_keep]
                options["datetime_column"] = None
                options["datetime_is_combined_year_month"] = False

        # Remove duplicates and null values
        initial_count = len(df)
        _console.print_debug(f"Starting with {initial_count} rows", tag="PREPROCESSING")

        # Check for null values before removing
        null_counts = df.isnull().sum()
        rows_with_nulls = df.isnull().any(axis=1).sum()
        if rows_with_nulls > 0:
            _console.print_debug(f"Found {rows_with_nulls} rows with null values", tag="NULL CHECK")
            for col, count in null_counts[null_counts > 0].items():
                _console.print_debug(f"  Column '{col}': {count} null values", tag="NULL CHECK")

        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates()
        after_dedup = len(df)
        duplicates_removed = before_dedup - after_dedup
        if duplicates_removed > 0:
            _console.print_debug(f"Removed {duplicates_removed} duplicate rows", tag="DUPLICATE REMOVAL")
            _console.print_debug(f"  Before: {before_dedup}, After: {after_dedup}", tag="DUPLICATE REMOVAL")

        # Remove null values
        before_dropna = len(df)
        df = df.dropna()
        after_dropna = len(df)
        nulls_removed = before_dropna - after_dropna
        if nulls_removed > 0:
            _console.print_debug(f"Removed {nulls_removed} rows with null values", tag="NULL REMOVAL")
            _console.print_debug(f"  Before: {before_dropna}, After: {after_dropna}", tag="NULL REMOVAL")

        # Summary
        total_removed = initial_count - len(df)
        percent_removed = (total_removed / initial_count * 100) if initial_count > 0 else 0
        _console.print_debug("Preprocessing complete:", tag="PREPROCESSING SUMMARY")
        _console.print_debug(f"  Initial rows: {initial_count}", tag="PREPROCESSING SUMMARY")
        _console.print_debug(f"  Final rows: {len(df)}", tag="PREPROCESSING SUMMARY")
        _console.print_debug(f"  Total removed: {total_removed} ({percent_removed:.1f}%)", tag="PREPROCESSING SUMMARY")
        _console.print_debug(f"  Retention rate: {(len(df) / initial_count * 100):.1f}%", tag="PREPROCESSING SUMMARY")

        if len(df) == 0:
            raise ValueError("No data remaining after removing duplicates and null values")

        if len(df) < initial_count * 0.1:
            _console.print_warning(
                f"Only {len(df)} rows remain from original {initial_count} after preprocessing"
            )

        _console.print_status(f"Preprocessed dataset has {len(df)} rows", "info")

        # Extract and normalize PageRank values if available
        if has_pagerank and pagerank_col in df.columns:
            import numpy as np

            pagerank_values = df[pagerank_col].values.astype(float)

            # Handle any NaN values (replace with min value)
            nan_mask = np.isnan(pagerank_values)
            if nan_mask.any():
                pagerank_values[nan_mask] = np.nanmin(pagerank_values)

            # Normalize to [0, 1] range then add 1 to get [1, 2] range
            pr_min, pr_max = pagerank_values.min(), pagerank_values.max()
            if pr_max > pr_min:
                normalized = (pagerank_values - pr_min) / (pr_max - pr_min)
            else:
                # All values are the same, set to 0.5 (middle of range)
                normalized = np.full_like(pagerank_values, 0.5)

            # Add 1 to get weights in [1, 2] range
            options["pagerank_weights"] = normalized + 1.0

            _console.print_debug(f"Extracted {len(pagerank_values)} PageRank weights", tag="PAGERANK")
            _console.print_debug(f"  Original range: [{pr_min:.6f}, {pr_max:.6f}]", tag="PAGERANK")
            _console.print_debug(
                f"  Weight range: [{options['pagerank_weights'].min():.4f}, {options['pagerank_weights'].max():.4f}]",
                tag="PAGERANK"
            )
            _console.print_status(f"PageRank weights extracted (range: 1.0-2.0)", "info")

            # Drop the pagerank column from df (no longer needed)
            df = df.drop(columns=[pagerank_col])
        else:
            options["pagerank_weights"] = None

        # Handle database persistence
        df = DatabaseManager.handle_dataframe_persistence(
            df, table_name, main_db_eng, save_to_db=options["save_to_db"]
        )

        return df
