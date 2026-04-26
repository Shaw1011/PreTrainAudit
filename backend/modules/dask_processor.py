"""
dask_processor.py — Dask-based out-of-core processing for massive datasets.

Provides chunked processing capabilities for datasets that exceed RAM.
Dask is optional - falls back to DuckDB/pandas if not available.
"""

from typing import Optional, Callable, Any
from pathlib import Path
import logging

logger = logging.getLogger("pretrainaudit")

# Try to import Dask, make it optional
DASK_AVAILABLE = False
try:
    import dask.dataframe as dd
    import dask.array as da
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    logger.info("Dask not available - falling back to DuckDB/pandas for large datasets")


class DaskConfig:
    """Configuration for Dask processing."""
    
    def __init__(
        self,
        n_workers: int = 4,
        memory_limit: str = "4GB",
        chunk_size: str = "256MB",
        enable_distributed: bool = False,
    ):
        self.n_workers = n_workers
        self.memory_limit = memory_limit
        self.chunk_size = chunk_size
        self.enable_distributed = enable_distributed
        self._client: Optional[Any] = None
    
    def get_client(self):
        """Get or create Dask client."""
        if not DASK_AVAILABLE:
            return None
        
        if self._client is None and self.enable_distributed:
            cluster = LocalCluster(
                n_workers=self.n_workers,
                memory_limit=self.memory_limit,
            )
            self._client = Client(cluster)
        
        return self._client
    
    def close(self):
        """Close Dask client if open."""
        if self._client is not None:
            self._client.close()
            self._client = None


def should_use_dask(file_size_bytes: int, threshold_gb: float = 1.0) -> bool:
    """Determine if Dask should be used based on file size.
    
    Args:
        file_size_bytes: Size of the file in bytes
        threshold_gb: Size threshold in GB above which Dask is recommended
    
    Returns:
        True if Dask should be used, False otherwise
    """
    if not DASK_AVAILABLE:
        return False
    
    threshold_bytes = threshold_gb * 1024 ** 3
    return file_size_bytes > threshold_bytes


def process_large_csv_with_dask(
    path: str,
    operations: list[Callable],
    sample_fraction: float = 0.1,
    config: Optional[DaskConfig] = None,
) -> dict:
    """Process a large CSV using Dask with chunked operations.
    
    Args:
        path: Path to CSV file
        operations: List of functions to apply to each chunk
        sample_fraction: Fraction of data to process (for speed)
        config: Dask configuration
    
    Returns:
        Aggregated results from all operations
    """
    if not DASK_AVAILABLE:
        logger.warning("Dask requested but not available - falling back to DuckDB")
        return {"error": "Dask not available", "fallback": "duckdb"}
    
    config = config or DaskConfig()
    
    try:
        # Read CSV with Dask
        ddf = dd.read_csv(path, blocksize=config.chunk_size)
        
        # Sample if requested
        if sample_fraction < 1.0:
            ddf = ddf.sample(frac=sample_fraction, random_state=42)
        
        # Compute basic stats
        total_rows = ddf.shape[0].compute()
        columns = list(ddf.columns)
        
        results = {
            "total_rows": total_rows,
            "columns": columns,
            "column_count": len(columns),
            "sampled_fraction": sample_fraction,
            "engine": "dask",
        }
        
        # Apply custom operations
        for op in operations:
            try:
                op_result = op(ddf).compute()
                results[f"custom_{op.__name__}"] = op_result
            except Exception as e:
                logger.error(f"Dask operation {op.__name__} failed: {e}")
                results[f"custom_{op.__name__}_error"] = str(e)
        
        return results
        
    except Exception as e:
        logger.error(f"Dask processing failed: {e}")
        return {"error": str(e), "engine": "dask"}


def compute_large_correlation_dask(
    path: str,
    numeric_columns: list[str],
    sample_fraction: float = 0.1,
    config: Optional[DaskConfig] = None,
) -> dict:
    """Compute correlation matrix for large datasets using Dask.
    
    Args:
        path: Path to CSV/Parquet file
        numeric_columns: List of numeric column names
        sample_fraction: Fraction to sample
    
    Returns:
        Correlation matrix as dict with columns and matrix
    """
    if not DASK_AVAILABLE:
        return {"error": "Dask not available"}
    
    config = config or DaskConfig()
    
    try:
        ext = Path(path).suffix.lower()
        
        if ext == ".csv":
            ddf = dd.read_csv(path, blocksize=config.chunk_size)
        elif ext == ".parquet":
            ddf = dd.read_parquet(path)
        else:
            return {"error": f"Unsupported format: {ext}"}
        
        # Select numeric columns
        ddf_numeric = ddf[numeric_columns]
        
        # Sample
        if sample_fraction < 1.0:
            ddf_numeric = ddf_numeric.sample(frac=sample_fraction, random_state=42)
        
        # Compute correlation
        corr = ddf_numeric.corr().compute()
        
        return {
            "columns": numeric_columns,
            "matrix": corr.values.tolist(),
            "engine": "dask",
            "sampled_fraction": sample_fraction,
        }
        
    except Exception as e:
        logger.error(f"Dask correlation failed: {e}")
        return {"error": str(e)}


def profile_large_dataset_dask(
    path: str,
    sample_size: int = 100_000,
    config: Optional[DaskConfig] = None,
) -> dict:
    """Profile a large dataset using Dask.
    
    Args:
        path: Path to dataset
        sample_size: Target sample size
        config: Dask configuration
    
    Returns:
        Profile results
    """
    if not DASK_AVAILABLE:
        return {"error": "Dask not available", "fallback": "duckdb"}
    
    config = config or DaskConfig()
    
    try:
        ext = Path(path).suffix.lower()
        
        if ext == ".csv":
            ddf = dd.read_csv(path, blocksize=config.chunk_size)
        elif ext == ".parquet":
            ddf = dd.read_parquet(path)
        else:
            return {"error": f"Unsupported format for Dask: {ext}"}
        
        # Get total rows (triggers compute)
        total_rows = ddf.shape[0].compute()
        
        # Calculate sample fraction
        sample_fraction = min(1.0, sample_size / max(total_rows, 1))
        
        # Sample
        ddf_sample = ddf.sample(frac=sample_fraction, random_state=42)
        df = ddf_sample.compute()
        
        # Profile the sample
        import numpy as np
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Missing values
        missing = df.isnull().sum().to_dict()
        missing_pct = {k: round(v / max(len(df), 1) * 100, 2) for k, v in missing.items()}
        
        # Duplicates
        dup_count = int(df.duplicated().sum())
        dup_pct = round(dup_count / max(len(df), 1) * 100, 2)
        
        return {
            "data_type": "tabular",
            "total_rows": total_rows,
            "sampled_rows": len(df),
            "sample_fraction": sample_fraction,
            "column_count": len(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_values": missing,
            "missing_pct": missing_pct,
            "duplicate_count": dup_count,
            "duplicate_pct": dup_pct,
            "engine": "dask",
        }
        
    except Exception as e:
        logger.error(f"Dask profiling failed: {e}")
        return {"error": str(e), "engine": "dask"}


# Convenience function to get appropriate engine
def get_processing_engine(
    file_size_bytes: int,
    threshold_gb: float = 1.0,
) -> str:
    """Get the appropriate processing engine for a file size.
    
    Returns:
        "dask" if file is large and Dask is available
        "duckdb" otherwise
    """
    if should_use_dask(file_size_bytes, threshold_gb):
        return "dask"
    return "duckdb"
