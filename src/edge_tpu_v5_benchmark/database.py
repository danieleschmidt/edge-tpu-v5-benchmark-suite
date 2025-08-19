"""Database operations and result persistence for benchmark data."""

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRecord:
    """Database record for a benchmark result."""
    id: Optional[int] = None
    timestamp: float = 0.0
    model_name: str = ""
    model_hash: str = ""
    device_info: str = ""
    benchmark_config: str = ""
    results_json: str = ""
    system_info: str = ""
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BenchmarkDatabase:
    """SQLite database for storing benchmark results and metadata."""

    def __init__(self, db_path: Union[str, Path] = "benchmark_results.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
        logger.info(f"Initialized benchmark database: {self.db_path}")

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Benchmark results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    model_hash TEXT NOT NULL,
                    device_info TEXT NOT NULL,
                    benchmark_config TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    system_info TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    INDEX(timestamp),
                    INDEX(model_name),
                    INDEX(model_hash)
                )
            """)

            # Model metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_hash TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    format TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    input_shape TEXT NOT NULL,
                    output_shape TEXT NOT NULL,
                    num_parameters INTEGER NOT NULL,
                    compilation_info TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    INDEX(model_hash),
                    INDEX(model_name)
                )
            """)

            # Device information table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS device_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_path TEXT NOT NULL,
                    tpu_version TEXT NOT NULL,
                    runtime_version TEXT NOT NULL,
                    compiler_version TEXT NOT NULL,
                    system_info TEXT NOT NULL,
                    last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(device_path, tpu_version, runtime_version)
                )
            """)

            # Performance statistics view
            conn.execute("""
                CREATE VIEW IF NOT EXISTS performance_summary AS
                SELECT 
                    model_name,
                    COUNT(*) as benchmark_count,
                    AVG(json_extract(results_json, '$.throughput')) as avg_throughput,
                    MIN(json_extract(results_json, '$.latency_metrics.p99')) as best_latency_p99,
                    MAX(json_extract(results_json, '$.power_metrics.efficiency')) as best_efficiency,
                    MAX(timestamp) as last_benchmarked
                FROM benchmark_results 
                GROUP BY model_name
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper locking."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def store_benchmark_result(
        self,
        model_name: str,
        model_hash: str,
        device_info: Dict[str, Any],
        benchmark_config: Dict[str, Any],
        results: Dict[str, Any],
        system_info: Dict[str, Any]
    ) -> int:
        """Store a benchmark result in the database.
        
        Args:
            model_name: Name of the benchmarked model
            model_hash: Hash of the model file
            device_info: TPU device information
            benchmark_config: Benchmark configuration parameters
            results: Benchmark results
            system_info: System information
            
        Returns:
            ID of the inserted record
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO benchmark_results 
                (timestamp, model_name, model_hash, device_info, benchmark_config, results_json, system_info)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                model_name,
                model_hash,
                json.dumps(device_info),
                json.dumps(benchmark_config),
                json.dumps(results),
                json.dumps(system_info)
            ))

            record_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Stored benchmark result for {model_name} with ID {record_id}")
            return record_id

    def store_model_metadata(self, metadata: Dict[str, Any]) -> int:
        """Store model metadata in the database.
        
        Args:
            metadata: Model metadata dictionary
            
        Returns:
            ID of the inserted record
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO model_metadata 
                (model_hash, model_name, model_path, format, size_bytes, 
                 input_shape, output_shape, num_parameters, compilation_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata["hash_sha256"],
                metadata["name"],
                metadata.get("path", ""),
                metadata["format"],
                metadata["size_bytes"],
                json.dumps(metadata["input_shape"]),
                json.dumps(metadata["output_shape"]),
                metadata["num_parameters"],
                json.dumps({
                    "optimization_level": metadata["optimization_level"],
                    "target_device": metadata["target_device"],
                    "compilation_time_seconds": metadata["compilation_time_seconds"],
                    "supported_ops_count": metadata["supported_ops_count"],
                    "unsupported_ops_count": metadata["unsupported_ops_count"]
                })
            ))

            record_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Stored model metadata for {metadata['name']} with ID {record_id}")
            return record_id

    def get_benchmark_results(
        self,
        model_name: Optional[str] = None,
        limit: int = 100,
        since_timestamp: Optional[float] = None
    ) -> List[BenchmarkRecord]:
        """Retrieve benchmark results with optional filtering.
        
        Args:
            model_name: Filter by model name
            limit: Maximum number of results
            since_timestamp: Only results after this timestamp
            
        Returns:
            List of benchmark records
        """
        query = "SELECT * FROM benchmark_results WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if since_timestamp:
            query += " AND timestamp > ?"
            params.append(since_timestamp)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []

            for row in cursor.fetchall():
                record = BenchmarkRecord(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    model_name=row["model_name"],
                    model_hash=row["model_hash"],
                    device_info=row["device_info"],
                    benchmark_config=row["benchmark_config"],
                    results_json=row["results_json"],
                    system_info=row["system_info"],
                    created_at=row["created_at"]
                )
                results.append(record)

            return results

    def get_performance_summary(self) -> List[Dict[str, Any]]:
        """Get performance summary for all models.
        
        Returns:
            List of performance summary records
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM performance_summary ORDER BY avg_throughput DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_model_history(self, model_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get benchmark history for a specific model.
        
        Args:
            model_name: Name of the model
            days: Number of days of history to retrieve
            
        Returns:
            List of benchmark results with parsed JSON
        """
        since_timestamp = time.time() - (days * 24 * 60 * 60)

        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT timestamp, results_json, benchmark_config
                FROM benchmark_results 
                WHERE model_name = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (model_name, since_timestamp))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "timestamp": row["timestamp"],
                    "results": json.loads(row["results_json"]),
                    "config": json.loads(row["benchmark_config"])
                })

            return results

    def cleanup_old_results(self, retention_days: int = 30) -> int:
        """Remove old benchmark results.
        
        Args:
            retention_days: Number of days to retain results
            
        Returns:
            Number of deleted records
        """
        cutoff_timestamp = time.time() - (retention_days * 24 * 60 * 60)

        with self._get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM benchmark_results 
                WHERE timestamp < ?
            """, (cutoff_timestamp,))

            deleted_count = cursor.rowcount
            conn.commit()

            logger.info(f"Cleaned up {deleted_count} old benchmark results")
            return deleted_count

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        with self._get_connection() as conn:
            stats = {}

            # Count of records in each table
            for table in ["benchmark_results", "model_metadata", "device_info"]:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Database file size
            stats["database_size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0
            stats["database_size_mb"] = stats["database_size_bytes"] / (1024 * 1024)

            # Date range of results
            cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM benchmark_results")
            min_ts, max_ts = cursor.fetchone()
            if min_ts and max_ts:
                stats["oldest_result"] = datetime.fromtimestamp(min_ts).isoformat()
                stats["newest_result"] = datetime.fromtimestamp(max_ts).isoformat()
                stats["time_span_days"] = (max_ts - min_ts) / (24 * 60 * 60)

            return stats

    def export_results(self, output_path: Union[str, Path], format: str = "json") -> None:
        """Export all benchmark results to a file.
        
        Args:
            output_path: Path to output file
            format: Export format ('json' or 'csv')
        """
        results = self.get_benchmark_results(limit=10000)  # Get all results

        if format.lower() == "json":
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_records": len(results),
                    "database_stats": self.get_database_stats()
                },
                "results": [result.to_dict() for result in results]
            }

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

        elif format.lower() == "csv":
            import csv

            with open(output_path, 'w', newline='') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
                    writer.writeheader()
                    for result in results:
                        writer.writerow(result.to_dict())

        logger.info(f"Exported {len(results)} results to {output_path}")


class ResultsCache:
    """In-memory cache for frequently accessed benchmark results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize results cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                return None

            # Check if expired
            if time.time() - self._timestamps[key] > self.ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                return None

            return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Remove oldest items if cache is full
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            self._cache[key] = value
            self._timestamps[key] = time.time()

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            expired_count = sum(1 for ts in self._timestamps.values()
                              if current_time - ts > self.ttl_seconds)

            return {
                "total_items": len(self._cache),
                "expired_items": expired_count,
                "valid_items": len(self._cache) - expired_count,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "memory_usage_estimate_mb": sum(
                    len(str(k)) + len(str(v)) for k, v in self._cache.items()
                ) / (1024 * 1024)
            }


class DataManager:
    """High-level data management interface combining database and cache."""

    def __init__(self, db_path: Optional[str] = None, cache_size: int = 1000):
        """Initialize data manager.
        
        Args:
            db_path: Database file path
            cache_size: Maximum cache size
        """
        self.database = BenchmarkDatabase(db_path or "benchmark_results.db")
        self.cache = ResultsCache(max_size=cache_size)
        logger.info("Initialized data manager with database and cache")

    def store_benchmark_result(self, model_name: str, model_hash: str,
                             device_info: Dict[str, Any], benchmark_config: Dict[str, Any],
                             results: Dict[str, Any], system_info: Dict[str, Any]) -> int:
        """Store benchmark result and invalidate related cache entries."""
        record_id = self.database.store_benchmark_result(
            model_name, model_hash, device_info, benchmark_config, results, system_info
        )

        # Invalidate related cache entries
        cache_keys_to_clear = [
            f"model_history_{model_name}",
            "performance_summary",
            f"model_results_{model_name}"
        ]

        for key in cache_keys_to_clear:
            if self.cache.get(key) is not None:
                self.cache._cache.pop(key, None)
                self.cache._timestamps.pop(key, None)

        return record_id

    def get_model_performance_history(
        self, model_name: str, days: int = 30, use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get model performance history with caching."""
        cache_key = f"model_history_{model_name}_{days}"

        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        results = self.database.get_model_history(model_name, days)

        if use_cache:
            self.cache.put(cache_key, results)

        return results

    def get_leaderboard_data(self, category: str = "all", use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get leaderboard data with caching."""
        cache_key = f"leaderboard_{category}"

        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Get performance summary from database
        summary = self.database.get_performance_summary()

        # Filter by category if specified
        if category != "all":
            # This is a simple filter - in practice you'd have more sophisticated categorization
            if category == "vision":
                summary = [s for s in summary if any(keyword in s["model_name"].lower()
                          for keyword in ["mobilenet", "efficientnet", "resnet", "yolo", "vit"])]
            elif category == "nlp":
                summary = [s for s in summary if any(keyword in s["model_name"].lower()
                          for keyword in ["bert", "gpt", "llama", "t5", "roberta"])]

        if use_cache:
            self.cache.put(cache_key, summary)

        return summary

    def cleanup_old_data(self, retention_days: int = 30) -> Dict[str, int]:
        """Cleanup old data and clear cache."""
        deleted_count = self.database.cleanup_old_results(retention_days)
        self.cache.clear()

        return {
            "deleted_records": deleted_count,
            "cache_cleared": True
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        db_stats = self.database.get_database_stats()
        cache_stats = self.cache.get_stats()

        return {
            "database": db_stats,
            "cache": cache_stats,
            "timestamp": datetime.now().isoformat()
        }

    def export_data(self, output_path: str, format: str = "json") -> None:
        """Export data to file."""
        self.database.export_results(output_path, format)
