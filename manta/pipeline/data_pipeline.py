"""
Data loading and preprocessing pipeline for MANTA topic analysis.
"""

import os
from typing import Dict, Any, Optional
import logging

import pandas as pd

from ..utils.console.console_manager import ConsoleManager
from ..utils.database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class DataPipeline:
    """Handles data loading and preprocessing operations."""
    
    @staticmethod
    def validate_inputs(filepath: Optional[str], desired_columns: str, options: Dict[str, Any],
                       dataframe: Optional[pd.DataFrame] = None) -> None:
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
        console: Optional[ConsoleManager] = None
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
        df = DataPipeline._apply_data_filters(dataframe.copy(), options, console)
        return df

    @staticmethod
    def load_data_file(filepath: str, options: Dict[str, Any], console: Optional[ConsoleManager] = None) -> pd.DataFrame:
        """
        Load data from CSV or Excel file.

        Args:
            filepath: Path to input file
            options: Configuration options containing separator and filter settings
            console: Console manager for status messages

        Returns:
            Loaded DataFrame
        """
        if console:
            console.print_status("Reading input file...", "processing")
        else:
            print("Reading input file...")

        if str(filepath).endswith(".csv"):
            # Read the CSV file with the specified separator
            df = pd.read_csv(
                filepath,
                encoding="utf-8",
                sep=options["separator"],
                engine="python",
                on_bad_lines="skip",
                #nrows=1_000, #TODO: will be removed when we have a better way to handle large files
            )
            #df = df.sample(n = 1_000)

            # Track initial data load
            initial_row_count = len(df)
            print(f"\n[DATA LOADING] Initial rows from CSV: {initial_row_count}")

            # Track year filter
            if "year" in df.columns:
                before_year_filter = len(df)
                df = df[df["year"]<2026]
                after_year_filter = len(df)
                rows_removed = before_year_filter - after_year_filter
                if rows_removed > 0:
                    print(f"[YEAR FILTER] Removed {rows_removed} rows with year >= 2026")
                    print(f"  Before: {before_year_filter} rows, After: {after_year_filter} rows")
            else:
                df = df[df["year"]<2026]
        elif str(filepath).endswith(".xlsx") or str(filepath).endswith(".xls"):
            df = pd.read_excel(filepath)

        # Apply data filters if specified
        df = DataPipeline._apply_data_filters(df, options, console)
        return df

    @staticmethod
    def _apply_data_filters(df: pd.DataFrame, options: Dict[str, Any], console: Optional[ConsoleManager] = None) -> pd.DataFrame:
        """Apply data filters based on configuration options."""
        try:
            if options.get("filter_app", False):
                initial_filter_count = len(df)
                print(f"\n[DATA FILTERS] Starting with {initial_filter_count} rows")

                filter_options = options.get("data_filter_options", {})
                if filter_options.get("filter_app_country", ""):
                    country_col = filter_options.get("filter_app_country_column", "")
                    if country_col in df.columns:
                        before_country = len(df)
                        df = df[df[country_col].str.upper() == filter_options["filter_app_country"]]
                        after_country = len(df)
                        removed = before_country - after_country
                        print(f"[COUNTRY FILTER] Removed {removed} rows")
                        print(f"  Filtered for country: {filter_options['filter_app_country']}")
                        print(f"  Before: {before_country}, After: {after_country}")
                        if console:
                            console.print_status(f"Applied country filter: {filter_options['filter_app_country']}", "info")
                    else:
                        msg = f"Warning: Filter column '{country_col}' not found in data"
                        if console:
                            console.print_status(msg, "warning")
                        else:
                            print(msg)

                if filter_options.get("filter_app_name", ""):
                    app_col = filter_options.get("filter_app_column", "")
                    if app_col in df.columns:
                        before_app = len(df)
                        df = df[df[app_col] == filter_options["filter_app_name"]]
                        after_app = len(df)
                        removed = before_app - after_app
                        print(f"[APP FILTER] Removed {removed} rows")
                        print(f"  Filtered for app: {filter_options['filter_app_name']}")
                        print(f"  Before: {before_app}, After: {after_app}")
                        if console:
                            console.print_status(f"Applied app filter: {filter_options['filter_app_name']}", "info")
                    else:
                        msg = f"Warning: Filter column '{app_col}' not found in data"
                        if console:
                            console.print_status(msg, "warning")
                        else:
                            print(msg)

                # Summary of filtering
                total_filtered = initial_filter_count - len(df)
                if total_filtered > 0:
                    print(f"[FILTERS SUMMARY] Total rows removed by filters: {total_filtered}")
        except KeyError as e:
            msg = f"Warning: Missing filter configuration: {e}"
            if console:
                console.print_status(msg, "warning")
            else:
                print(msg)
        except Exception as e:
            msg = f"Warning: Error applying data filters: {e}"
            if console:
                console.print_status(msg, "warning")
            else:
                print(msg)

        return df

    @staticmethod
    def preprocess_dataframe(
        df: pd.DataFrame,
        desired_columns: str,
        options: Dict[str, Any],
        main_db_eng,
        table_name: str,
        console: Optional[ConsoleManager] = None
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
        if console:
            console.print_status("Preprocessing data...", "processing")

        # Select only desired columns and validate they exist
        if desired_columns not in df.columns:
            available_columns = ", ".join(df.columns.tolist())
            raise KeyError(f"Column '{desired_columns}' not found in data. Available columns: {available_columns}")

        # Check for datetime columns to preserve for temporal analysis
        common_datetime_cols = ['year', 'date', 'datetime', 'timestamp', 'time',
                                'rev_submit_millis_since_epoch', 'created_at', 'updated_at']
        datetime_col_found = None

        # Check if both 'year' and 'month' columns exist - combine them into MM-YYYY format
        if 'year' in df.columns and 'month' in df.columns:
            if console:
                console.print_status("Combining 'year' and 'month' columns into datetime...", "processing")

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
                if month_str is None or month_str == 'None':
                    logger.warning(f"Invalid month value encountered: {month_val}. Defaulting to January.")
                    return 1

                month_map = {
                    'jan': 1, 'january': 1,
                    'feb': 2, 'february': 2,
                    'mar': 3, 'march': 3,
                    'apr': 4, 'april': 4,
                    'may': 5,
                    'jun': 6, 'june': 6,
                    'jul': 7, 'july': 7,
                    'aug': 8, 'august': 8,
                    'sep': 9, 'sept': 9, 'september': 9,
                    'oct': 10, 'october': 10,
                    'nov': 11, 'november': 11,
                    'dec': 12, 'december': 12
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
            df['month_numeric'] = df['month'].apply(convert_month_to_numeric)

            # Combine year and month into a datetime column (day=1 for proper sorting)
            df['datetime_combined'] = pd.to_datetime(
                df['year'].astype(int).astype(str) + '-' +
                df['month_numeric'].astype(int).astype(str).str.zfill(2) + '-01',
                format='%Y-%m-%d',
                errors='coerce'
            )

            # Store that we combined year and month
            datetime_col_found = 'datetime_combined'
            options['datetime_column'] = datetime_col_found
            options['datetime_is_combined_year_month'] = True

            # Select columns to keep
            df = df[[desired_columns, 'datetime_combined']]

            if console:
                console.print_status(f"Created combined datetime column from year and month", "info")
        else:
            # Original logic for other datetime columns
            for col in common_datetime_cols:
                if col in df.columns:
                    datetime_col_found = col
                    break

            # Select text column and datetime column if available
            if datetime_col_found:
                df = df[[desired_columns, datetime_col_found]]
                options['datetime_column'] = datetime_col_found  # Store for later use in pipeline
                options['datetime_is_combined_year_month'] = False
                if console:
                    console.print_status(f"Preserved datetime column: {datetime_col_found}", "info")
            else:
                df = df[[desired_columns]]
                options['datetime_column'] = None
                options['datetime_is_combined_year_month'] = False
        
        # Remove duplicates and null values
        initial_count = len(df)
        print(f"\n[PREPROCESSING] Starting with {initial_count} rows")

        # Check for null values before removing
        null_counts = df.isnull().sum()
        rows_with_nulls = df.isnull().any(axis=1).sum()
        if rows_with_nulls > 0:
            print(f"[NULL CHECK] Found {rows_with_nulls} rows with null values")
            for col, count in null_counts[null_counts > 0].items():
                print(f"  Column '{col}': {count} null values")

        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates()
        after_dedup = len(df)
        duplicates_removed = before_dedup - after_dedup
        if duplicates_removed > 0:
            print(f"[DUPLICATE REMOVAL] Removed {duplicates_removed} duplicate rows")
            print(f"  Before: {before_dedup}, After: {after_dedup}")

        # Remove null values
        before_dropna = len(df)
        df = df.dropna()
        after_dropna = len(df)
        nulls_removed = before_dropna - after_dropna
        if nulls_removed > 0:
            print(f"[NULL REMOVAL] Removed {nulls_removed} rows with null values")
            print(f"  Before: {before_dropna}, After: {after_dropna}")

        # Summary
        total_removed = initial_count - len(df)
        percent_removed = (total_removed / initial_count * 100) if initial_count > 0 else 0
        print(f"\n[PREPROCESSING SUMMARY]")
        print(f"  Initial rows: {initial_count}")
        print(f"  Final rows: {len(df)}")
        print(f"  Total removed: {total_removed} ({percent_removed:.1f}%)")
        print(f"  Retention rate: {(len(df)/initial_count*100):.1f}%")

        if len(df) == 0:
            raise ValueError("No data remaining after removing duplicates and null values")

        if len(df) < initial_count * 0.1:
            msg = f"Warning: Only {len(df)} rows remain from original {initial_count} after preprocessing"
            if console:
                console.print_status(msg, "warning")
            else:
                print(msg)

        msg = f"Preprocessed dataset has {len(df)} rows"
        if console:
            console.print_status(msg, "info")
        else:
            print(f"File has {len(df)} rows.")

        # Handle database persistence
        df = DatabaseManager.handle_dataframe_persistence(
            df, table_name, main_db_eng, save_to_db=options["save_to_db"]
        )
        
        return df
