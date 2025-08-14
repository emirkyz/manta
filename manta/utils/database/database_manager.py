
"""
Database management utilities for MANTA topic modeling.

This module provides utilities for database initialization, directory setup,
and database engine management for the MANTA topic modeling pipeline.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, Engine, MetaData, Table, Column, String, insert, text


@dataclass
class DatabaseConfig:
    """Configuration object for database setup."""
    
    program_output_dir: Path
    instance_path: Path
    output_dir: Path
    topics_db_engine: Engine
    main_db_engine: Engine


class DatabaseManager:
    """Manages database initialization and configuration for MANTA processing."""
    
    @staticmethod
    def setup_directories(output_base_dir: Optional[str] = None) -> Tuple[Path, Path, Path]:
        """
        Set up necessary directories for MANTA processing.
        
        Args:
            output_base_dir: Base directory for outputs. If None, uses current working directory.
            
        Returns:
            Tuple of (program_output_dir, instance_path, output_dir)
        """
        if output_base_dir is None:
            base_dir = Path.cwd()
        else:
            base_dir = Path(output_base_dir).resolve()

        program_output_dir = base_dir / "TopicAnalysis"
        instance_path = program_output_dir / "instance"
        output_dir = program_output_dir / "Output"

        # Create necessary directories
        instance_path.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return program_output_dir, instance_path, output_dir
    
    @staticmethod
    def create_database_engines(instance_path: Path) -> Tuple[Engine, Engine]:
        """
        Create SQLAlchemy database engines for topics and main data storage.
        
        Args:
            instance_path: Path to the instance directory where databases are stored.
            
        Returns:
            Tuple of (topics_db_engine, main_db_engine)
        """
        print("Using new database engines...")
        topics_db_engine = create_engine(
            f'sqlite:///{instance_path / "topics.db"}'
        )
        main_db_engine = create_engine(
            f'sqlite:///{instance_path / "scopus.db"}'
        )
        
        return topics_db_engine, main_db_engine
    
    @classmethod
    def initialize_database_config(cls, output_base_dir: Optional[str] = None) -> DatabaseConfig:
        """
        Initialize complete database configuration including directories and engines.
        
        Args:
            output_base_dir: Base directory for outputs. If None, uses current working directory.
            
        Returns:
            DatabaseConfig object with all necessary paths and engines.
        """
        program_output_dir, instance_path, output_dir = cls.setup_directories(output_base_dir)
        topics_db_engine, main_db_engine = cls.create_database_engines(instance_path)
        
        return DatabaseConfig(
            program_output_dir=program_output_dir,
            instance_path=instance_path,
            output_dir=output_dir,
            topics_db_engine=topics_db_engine,
            main_db_engine=main_db_engine
        )
    
    @staticmethod
    def table_exists(engine: Engine, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            existing_tables = pd.read_sql_query(tables_query, engine, params=(table_name,))
            return len(existing_tables) > 0
        except Exception as e:
            print(f"Error checking table existence: {e}")
            return False
    
    @staticmethod
    def save_dataframe_to_db(
        df: pd.DataFrame, 
        table_name: str, 
        engine: Engine, 
        if_exists: str = "replace"
    ) -> bool:
        """
        Save DataFrame to database table.
        
        Args:
            df: DataFrame to save
            table_name: Name of the target table
            engine: SQLAlchemy engine
            if_exists: Action if table exists ('replace', 'append', 'fail')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Saving DataFrame to database table: {table_name}")
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)
            return True
        except Exception as e:
            print(f"Error saving DataFrame to database: {e}")
            return False
    
    @staticmethod
    def load_dataframe_from_db(table_name: str, engine: Engine) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from database table.
        
        Args:
            table_name: Name of the table to load
            engine: SQLAlchemy engine
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            print(f"Loading DataFrame from database table: {table_name}")
            return pd.read_sql_table(table_name, engine)
        except Exception as e:
            print(f"Error loading DataFrame from database: {e}")
            return None
    
    @staticmethod
    def save_topics_to_database(
        topics_data: Dict[str, List[str]], 
        data_frame_name: str, 
        topics_db_engine: Engine
    ) -> bool:
        """
        Save topics to database using SQLAlchemy engine.
        
        Args:
            topics_data: Dictionary containing topic names as keys and lists of words as values
            data_frame_name: Name of the data frame
            topics_db_engine: SQLAlchemy engine for topics database
            
        Returns:
            True if successful, False otherwise
        """
        if not topics_db_engine:
            print("Warning: No database engine provided, skipping database save")
            return False
            
        try:
            print(f"Saving topics to database for: {data_frame_name}")
            
            # Create metadata
            metadata = MetaData()
            table_name = f"{data_frame_name}_topics"
            
            # Drop table if it exists
            with topics_db_engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                
            # Create table with dynamic columns based on topics
            columns = [Column(topic, String) for topic in topics_data.keys()]
            table = Table(table_name, metadata, *columns)
            
            # Create table
            metadata.create_all(topics_db_engine)

            # Prepare data for insertion
            max_words = max(len(words) for words in topics_data.values())
            rows = []
            for i in range(max_words):
                row = {}
                for topic, words in topics_data.items():
                    row[topic] = words[i] if i < len(words) else None
                rows.append(row)

            # Insert data using SQLAlchemy
            with topics_db_engine.begin() as conn:
                if rows:  # Only insert if we have data
                    conn.execute(insert(table), rows)
                    
            return True
        except Exception as e:
            print(f"Error saving topics to database: {e}")
            return False
    
    @classmethod
    def handle_dataframe_persistence(
        cls,
        df: pd.DataFrame,
        table_name: str,
        engine: Engine,
        save_to_db: bool = False
    ) -> pd.DataFrame:
        """
        Handle DataFrame persistence to/from database based on configuration.
        
        Args:
            df: DataFrame to process
            table_name: Database table name
            engine: SQLAlchemy engine
            save_to_db: Whether to save to database
            
        Returns:
            DataFrame (either original or loaded from database)
        """
        if save_to_db:
            print("Adding data to main database...")
            
            # Save to database (always replace if exists for simplicity)
            success = cls.save_dataframe_to_db(df, table_name, engine, if_exists="replace")
            
            if success:
                # Load back from database
                loaded_df = cls.load_dataframe_from_db(table_name, engine)
                if loaded_df is not None:
                    return loaded_df
                else:
                    print("Warning: Failed to load data back from database, using original DataFrame")
                    return df
            else:
                print("Warning: Failed to save to database, using original DataFrame")
                return df
        else:
            print("Not saving data to main database...")
            return pd.DataFrame(df)
    
    @staticmethod
    def execute_query(query: str, engine: Engine, params: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Execute a SQL query and return the results as a DataFrame.
        
        Args:
            query: SQL query to execute
            engine: SQLAlchemy engine
            params: Optional parameters for parameterized queries
            
        Returns:
            DataFrame with query results, None if error occurred
        """
        try:
            return pd.read_sql_query(query, engine, params=params)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None