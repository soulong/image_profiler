"""SQLite database operations for saving analysis results."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class Database:
    """SQLite database for saving analysis results.
    
    Parameters
    ----------
    db_path : str or Path
        Path to SQLite database file.
    """
    
    def __init__(self, db_path: Union[str, Path]):
        """Initialize Database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def connect(self) -> None:
        """Connect to database."""
        self._conn = sqlite3.connect(self.db_path)
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def save_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace"
    ) -> None:
        """Save DataFrame to database table.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save.
        table_name : str
            Name of the table.
        if_exists : str
            Behavior if table exists: "replace", "append", or "fail".
        """
        if self._conn is None:
            raise RuntimeError("Database not connected. Use connect() or context manager.")
        
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == "object":
                if any(isinstance(x, Path) for x in df_copy[col] if pd.notna(x)):
                    df_copy[col] = df_copy[col].astype(str)
        
        df_copy.to_sql(table_name, self._conn, if_exists=if_exists, index=False)
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results.
        
        Parameters
        ----------
        sql : str
            SQL query.
            
        Returns
        -------
        pd.DataFrame
            Query results.
        """
        if self._conn is None:
            raise RuntimeError("Database not connected. Use connect() or context manager.")
        
        return pd.read_sql_query(sql, self._conn)
    
    def get_tables(self) -> list:
        """Get list of tables in database.
        
        Returns
        -------
        list
            List of table names.
        """
        if self._conn is None:
            raise RuntimeError("Database not connected. Use connect() or context manager.")
        
        cursor = self._conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]


def write_results_to_db(
    db_path: Path,
    table_name: str,
    results: pd.DataFrame,
    if_exists: str = "append"
) -> None:
    """Write profiling results to SQLite database.
    
    Parameters
    ----------
    db_path : Path
        Path to SQLite database.
    table_name : str
        Table name (e.g., "image", "cell", "nuclei").
    results : pd.DataFrame
        Results DataFrame to write.
    if_exists : str
        "append", "replace", or "fail".
    """
    with Database(db_path) as db:
        db.save_table(results, table_name, if_exists=if_exists)


def save_metadata_to_db(
    db_path: Path,
    metadata: pd.DataFrame,
    if_exists: str = "replace"
) -> None:
    """Save metadata to SQLite database.
    
    Parameters
    ----------
    db_path : Path
        Path to SQLite database.
    metadata : pd.DataFrame
        Metadata DataFrame to save.
    if_exists : str
        "append", "replace", or "fail".
    """
    write_results_to_db(db_path, "metadata", metadata, if_exists=if_exists)
