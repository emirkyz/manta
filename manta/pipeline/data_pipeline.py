"""
Data loading and preprocessing pipeline for MANTA topic analysis.
"""

import os
from typing import Dict, Any, Optional

import pandas as pd

from ..utils.console.console_manager import ConsoleManager
from ..utils.database.database_manager import DatabaseManager


class DataPipeline:
    """Handles data loading and preprocessing operations."""
    
    @staticmethod
    def validate_inputs(filepath: str, desired_columns: str, options: Dict[str, Any]) -> None:
        """
        Validate input parameters for processing.
        
        Args:
            filepath: Path to input file
            desired_columns: Column name containing text data
            options: Configuration options
            
        Raises:
            ValueError: If inputs are invalid
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")
        
        if not desired_columns or not desired_columns.strip():
            raise ValueError("desired_columns cannot be empty")
            
        required_options = ["LANGUAGE", "DESIRED_TOPIC_COUNT", "N_TOPICS"]
        for option in required_options:
            if option not in options:
                raise ValueError(f"Missing required option: {option}")
        
        if options["LANGUAGE"] not in ["TR", "EN"]:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}. Must be 'TR' or 'EN'")

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
                nrows=100_000, #TODO: will be removed when we have a better way to handle large files
            )

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
                filter_options = options.get("data_filter_options", {})
                if filter_options.get("filter_app_country", ""):
                    country_col = filter_options.get("filter_app_country_column", "")
                    if country_col in df.columns:
                        df = df[df[country_col].str.upper() == filter_options["filter_app_country"]]
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
                        df = df[df[app_col] == filter_options["filter_app_name"]]
                        if console:
                            console.print_status(f"Applied app filter: {filter_options['filter_app_name']}", "info")
                    else:
                        msg = f"Warning: Filter column '{app_col}' not found in data"
                        if console:
                            console.print_status(msg, "warning")
                        else:
                            print(msg)
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
        
        df = df[desired_columns]
        
        # Remove duplicates and null values
        initial_count = len(df)
        df = df.drop_duplicates()
        df = df.dropna()
        
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
