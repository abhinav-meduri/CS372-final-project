"""
Data Loader Module
Handles loading and parsing PatentsView TSV files from zip archives.
"""

import zipfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Generator, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatentDataLoader:
    """Load patent data from PatentsView TSV zip files."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.years = [2021, 2022, 2023, 2024, 2025]
    
    def get_file_path(self, year: int, file_type: str) -> Path:
        """Get the path to a specific data file."""
        year_dir = self.data_dir / str(year)
        
        if file_type == "claims":
            # Handle potential naming variations
            candidates = [
                year_dir / f"g_claims_{year}.tsv.zip",
                year_dir / f"g_claims_{year}.tsv (1).zip",
            ]
        elif file_type == "summary":
            candidates = [
                year_dir / f"g_brf_sum_text_{year}.tsv.zip",
                year_dir / f"g_brf_sum_text_{year}.tsv (1).zip",
            ]
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
        for path in candidates:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"No {file_type} file found for year {year}")
    
    def load_tsv_from_zip(
        self, 
        zip_path: Path, 
        chunksize: Optional[int] = None
    ) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Load a TSV file from a zip archive.
        
        Args:
            zip_path: Path to the zip file
            chunksize: If provided, return a generator of chunks
            
        Returns:
            DataFrame or generator of DataFrames
        """
        logger.info(f"Loading {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get the TSV file name inside the zip
            tsv_files = [f for f in zf.namelist() if f.endswith('.tsv')]
            if not tsv_files:
                raise ValueError(f"No TSV file found in {zip_path}")
            
            tsv_name = tsv_files[0]
            logger.info(f"  Found TSV: {tsv_name}")
            
            with zf.open(tsv_name) as tsv_file:
                if chunksize:
                    return pd.read_csv(
                        tsv_file, 
                        sep='\t', 
                        chunksize=chunksize,
                        low_memory=False,
                        on_bad_lines='skip'
                    )
                else:
                    return pd.read_csv(
                        tsv_file, 
                        sep='\t',
                        low_memory=False,
                        on_bad_lines='skip'
                    )
    
    def load_claims(self, year: int, chunksize: Optional[int] = None) -> pd.DataFrame:
        """Load claims data for a specific year."""
        path = self.get_file_path(year, "claims")
        return self.load_tsv_from_zip(path, chunksize)
    
    def load_summary(self, year: int, chunksize: Optional[int] = None) -> pd.DataFrame:
        """Load brief summary text data for a specific year."""
        path = self.get_file_path(year, "summary")
        return self.load_tsv_from_zip(path, chunksize)
    
    def load_year_data(self, year: int) -> Dict[str, pd.DataFrame]:
        """Load all data for a specific year."""
        return {
            "claims": self.load_claims(year),
            "summary": self.load_summary(year)
        }
    
    def get_available_years(self) -> List[int]:
        """Get list of years with available data."""
        available = []
        for year in self.years:
            try:
                self.get_file_path(year, "claims")
                self.get_file_path(year, "summary")
                available.append(year)
            except FileNotFoundError:
                continue
        return available
    
    def peek_columns(self, year: int, file_type: str, nrows: int = 5) -> pd.DataFrame:
        """Peek at the first few rows to understand schema."""
        path = self.get_file_path(year, file_type)
        
        with zipfile.ZipFile(path, 'r') as zf:
            tsv_files = [f for f in zf.namelist() if f.endswith('.tsv')]
            tsv_name = tsv_files[0]
            
            with zf.open(tsv_name) as tsv_file:
                return pd.read_csv(tsv_file, sep='\t', nrows=nrows, low_memory=False)


def explore_data_schema():
    """Utility function to explore the data schema."""
    loader = PatentDataLoader()
    available_years = loader.get_available_years()
    
    print(f"Available years: {available_years}")
    
    if available_years:
        year = available_years[0]
        print(f"\n=== Claims Schema ({year}) ===")
        claims_sample = loader.peek_columns(year, "claims")
        print(f"Columns: {list(claims_sample.columns)}")
        print(claims_sample.head())
        
        print(f"\n=== Summary Schema ({year}) ===")
        summary_sample = loader.peek_columns(year, "summary")
        print(f"Columns: {list(summary_sample.columns)}")
        print(summary_sample.head())


if __name__ == "__main__":
    explore_data_schema()


