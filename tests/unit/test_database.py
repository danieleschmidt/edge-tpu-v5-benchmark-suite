"""Tests for database module."""

import pytest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

from edge_tpu_v5_benchmark.database import (
    BenchmarkDatabase, 
    BenchmarkRecord, 
    DataManager, 
    ResultsCache
)


class TestBenchmarkDatabase:
    """Test cases for BenchmarkDatabase class."""
    
    def test_init_creates_database(self, temp_dir):
        """Test database initialization creates tables."""
        db_path = temp_dir / "test.db"
        db = BenchmarkDatabase(db_path)
        
        assert db.db_path.exists()
        
        # Check that tables exist
        with db._get_connection() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('benchmark_results', 'model_metadata', 'device_info')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
        assert "benchmark_results" in tables
        assert "model_metadata" in tables
        assert "device_info" in tables
    
    def test_store_benchmark_result(self, temp_dir):
        """Test storing a benchmark result."""
        db = BenchmarkDatabase(temp_dir / "test.db")
        
        model_name = "test_model"
        model_hash = "abc123"
        device_info = {"device": "mock_tpu", "version": "v5"}
        benchmark_config = {"iterations": 100, "warmup": 10}
        results = {"throughput": 100.0, "latency": 0.01}
        system_info = {"cpu": "test_cpu", "memory": "8GB"}
        
        record_id = db.store_benchmark_result(
            model_name, model_hash, device_info, 
            benchmark_config, results, system_info
        )
        
        assert record_id > 0
        
        # Verify the record was stored
        records = db.get_benchmark_results(limit=1)
        assert len(records) == 1
        assert records[0].model_name == model_name
        assert records[0].model_hash == model_hash
    
    def test_store_model_metadata(self, temp_dir):
        """Test storing model metadata."""
        db = BenchmarkDatabase(temp_dir / "test.db")
        
        metadata = {
            "hash_sha256": "def456",
            "name": "test_model",
            "path": "/path/to/model",
            "format": "onnx",
            "size_bytes": 1000000,
            "input_shape": [1, 3, 224, 224],
            "output_shape": [1, 1000],
            "num_parameters": 500000,
            "optimization_level": 3,
            "target_device": "tpu_v5_edge",
            "compilation_time_seconds": 5.0,
            "supported_ops_count": 50,
            "unsupported_ops_count": 2
        }
        
        record_id = db.store_model_metadata(metadata)
        assert record_id > 0
    
    def test_get_benchmark_results_with_filter(self, temp_dir):
        """Test retrieving benchmark results with filtering."""
        db = BenchmarkDatabase(temp_dir / "test.db")
        
        # Store multiple results
        for i in range(5):
            db.store_benchmark_result(
                f"model_{i}", 
                f"hash_{i}",
                {"device": "mock_tpu"},
                {"iterations": 100},
                {"throughput": float(i * 10)},
                {"system": "test"}
            )
        
        # Test filtering by model name
        results = db.get_benchmark_results(model_name="model_2")
        assert len(results) == 1
        assert results[0].model_name == "model_2"
        
        # Test limit
        results = db.get_benchmark_results(limit=3)
        assert len(results) == 3
    
    def test_get_performance_summary(self, temp_dir):
        """Test performance summary generation."""
        db = BenchmarkDatabase(temp_dir / "test.db")
        
        # Store results for different models
        models = ["mobilenet", "resnet", "mobilenet"]
        throughputs = [100.0, 80.0, 95.0]
        
        for model, throughput in zip(models, throughputs):
            db.store_benchmark_result(
                model, f"hash_{model}",
                {"device": "mock_tpu"},
                {"iterations": 100},
                {"throughput": throughput, "latency_metrics": {"p99": 0.01}},
                {"system": "test"}
            )
        
        summary = db.get_performance_summary()
        assert len(summary) == 2  # mobilenet and resnet
        
        # Check that mobilenet has 2 benchmarks
        mobilenet_summary = next(s for s in summary if s["model_name"] == "mobilenet")
        assert mobilenet_summary["benchmark_count"] == 2
    
    def test_cleanup_old_results(self, temp_dir):
        """Test cleanup of old benchmark results."""
        db = BenchmarkDatabase(temp_dir / "test.db")
        
        # Store some results
        for i in range(3):
            db.store_benchmark_result(
                f"model_{i}", f"hash_{i}",
                {"device": "mock_tpu"},
                {"iterations": 100},
                {"throughput": 100.0},
                {"system": "test"}
            )
        
        # Manually update timestamps to make them old
        old_timestamp = time.time() - (40 * 24 * 60 * 60)  # 40 days ago
        
        with db._get_connection() as conn:
            conn.execute(
                "UPDATE benchmark_results SET timestamp = ? WHERE model_name = ?",
                (old_timestamp, "model_0")
            )
            conn.commit()
        
        # Cleanup results older than 30 days
        deleted_count = db.cleanup_old_results(retention_days=30)
        assert deleted_count == 1
        
        # Verify only 2 results remain
        remaining_results = db.get_benchmark_results()
        assert len(remaining_results) == 2
        assert all(r.model_name != "model_0" for r in remaining_results)
    
    def test_get_database_stats(self, temp_dir):
        """Test database statistics retrieval."""
        db = BenchmarkDatabase(temp_dir / "test.db")
        
        # Store some data
        db.store_benchmark_result(
            "test_model", "test_hash",
            {"device": "mock_tpu"},
            {"iterations": 100},
            {"throughput": 100.0},
            {"system": "test"}
        )
        
        stats = db.get_database_stats()
        
        assert "benchmark_results_count" in stats
        assert "model_metadata_count" in stats
        assert "device_info_count" in stats
        assert "database_size_bytes" in stats
        assert "database_size_mb" in stats
        
        assert stats["benchmark_results_count"] == 1
        assert stats["database_size_bytes"] > 0
    
    def test_export_results_json(self, temp_dir):
        """Test exporting results to JSON."""
        db = BenchmarkDatabase(temp_dir / "test.db")
        
        # Store a result
        db.store_benchmark_result(
            "test_model", "test_hash",
            {"device": "mock_tpu"},
            {"iterations": 100},
            {"throughput": 100.0},
            {"system": "test"}
        )
        
        export_path = temp_dir / "export.json"
        db.export_results(export_path, format="json")
        
        assert export_path.exists()
        
        # Verify export content
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["model_name"] == "test_model"


class TestResultsCache:
    """Test cases for ResultsCache class."""
    
    def test_cache_put_get(self):
        """Test basic cache put/get operations."""
        cache = ResultsCache(max_size=10, ttl_seconds=60)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
    
    def test_cache_expiration(self):
        """Test cache item expiration."""
        cache = ResultsCache(max_size=10, ttl_seconds=1)  # 1 second TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Sleep to let item expire
        time.sleep(1.1)
        assert cache.get("key1") is None
    
    def test_cache_size_limit(self):
        """Test cache size limiting."""
        cache = ResultsCache(max_size=2, ttl_seconds=60)
        
        # Fill cache to capacity
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert cache.size() == 2
        
        # Add another item (should evict oldest)
        cache.put("key3", "value3")
        assert cache.size() == 2
        assert cache.get("key1") is None  # Oldest item evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = ResultsCache(max_size=10, ttl_seconds=60)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert cache.size() == 2
        
        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ResultsCache(max_size=10, ttl_seconds=1)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Let one item expire
        time.sleep(1.1)
        cache.put("key3", "value3")  # This will clean up expired items
        
        stats = cache.get_stats()
        assert "total_items" in stats
        assert "expired_items" in stats
        assert "valid_items" in stats
        assert "max_size" in stats
        assert "ttl_seconds" in stats
        assert "memory_usage_estimate_mb" in stats


class TestDataManager:
    """Test cases for DataManager class."""
    
    def test_init(self, temp_dir):
        """Test DataManager initialization."""
        db_path = str(temp_dir / "test.db")
        manager = DataManager(db_path=db_path, cache_size=100)
        
        assert manager.database.db_path.exists()
        assert manager.cache.max_size == 100
    
    def test_store_benchmark_result_invalidates_cache(self, temp_dir):
        """Test that storing results invalidates related cache entries."""
        db_path = str(temp_dir / "test.db")
        manager = DataManager(db_path=db_path)
        
        # Put something in cache
        manager.cache.put("model_history_test_model", ["cached_data"])
        manager.cache.put("performance_summary", ["cached_summary"])
        
        assert manager.cache.get("model_history_test_model") is not None
        assert manager.cache.get("performance_summary") is not None
        
        # Store a benchmark result (should invalidate cache)
        manager.store_benchmark_result(
            "test_model", "hash123",
            {"device": "mock_tpu"},
            {"iterations": 100},
            {"throughput": 100.0},
            {"system": "test"}
        )
        
        # Cache entries should be invalidated
        assert manager.cache.get("model_history_test_model") is None
        assert manager.cache.get("performance_summary") is None
    
    def test_get_model_performance_history_with_cache(self, temp_dir):
        """Test getting model performance history with caching."""
        db_path = str(temp_dir / "test.db")
        manager = DataManager(db_path=db_path)
        
        # Store some benchmark results
        for i in range(3):
            manager.store_benchmark_result(
                "test_model", f"hash_{i}",
                {"device": "mock_tpu"},
                {"iterations": 100},
                {"throughput": float(100 + i)},
                {"system": "test"}
            )
        
        # First call should hit database and cache result
        history1 = manager.get_model_performance_history("test_model", days=30, use_cache=True)
        assert len(history1) == 3
        
        # Second call should hit cache
        with patch.object(manager.database, 'get_model_history') as mock_db:
            history2 = manager.get_model_performance_history("test_model", days=30, use_cache=True)
            mock_db.assert_not_called()  # Should not hit database
            assert history2 == history1
    
    def test_get_leaderboard_data_filtering(self, temp_dir):
        """Test leaderboard data filtering by category."""
        db_path = str(temp_dir / "test.db")
        manager = DataManager(db_path=db_path)
        
        # Mock the database performance summary
        mock_summary = [
            {"model_name": "mobilenet_v3", "avg_throughput": 100},
            {"model_name": "bert_base", "avg_throughput": 80},
            {"model_name": "resnet_50", "avg_throughput": 90},
            {"model_name": "gpt2_small", "avg_throughput": 60},
        ]
        
        with patch.object(manager.database, 'get_performance_summary', return_value=mock_summary):
            # Test vision category filtering
            vision_data = manager.get_leaderboard_data(category="vision", use_cache=False)
            vision_models = [item["model_name"] for item in vision_data]
            assert "mobilenet_v3" in vision_models
            assert "resnet_50" in vision_models
            assert "bert_base" not in vision_models
            assert "gpt2_small" not in vision_models
            
            # Test NLP category filtering
            nlp_data = manager.get_leaderboard_data(category="nlp", use_cache=False)
            nlp_models = [item["model_name"] for item in nlp_data]
            assert "bert_base" in nlp_models
            assert "gpt2_small" in nlp_models
            assert "mobilenet_v3" not in nlp_models
            assert "resnet_50" not in nlp_models
            
            # Test all category (no filtering)
            all_data = manager.get_leaderboard_data(category="all", use_cache=False)
            assert len(all_data) == 4
    
    def test_cleanup_old_data(self, temp_dir):
        """Test cleanup of old data and cache clearing."""
        db_path = str(temp_dir / "test.db")
        manager = DataManager(db_path=db_path)
        
        # Put something in cache
        manager.cache.put("test_key", "test_value")
        assert manager.cache.get("test_key") is not None
        
        # Mock database cleanup
        with patch.object(manager.database, 'cleanup_old_results', return_value=5) as mock_cleanup:
            result = manager.cleanup_old_data(retention_days=30)
            
            mock_cleanup.assert_called_once_with(30)
            assert result["deleted_records"] == 5
            assert result["cache_cleared"] is True
            
            # Cache should be cleared
            assert manager.cache.get("test_key") is None
    
    def test_get_system_stats(self, temp_dir):
        """Test getting comprehensive system statistics."""
        db_path = str(temp_dir / "test.db")
        manager = DataManager(db_path=db_path)
        
        stats = manager.get_system_stats()
        
        assert "database" in stats
        assert "cache" in stats
        assert "timestamp" in stats
        
        # Check database stats structure
        db_stats = stats["database"]
        assert "benchmark_results_count" in db_stats
        assert "database_size_bytes" in db_stats
        
        # Check cache stats structure
        cache_stats = stats["cache"]
        assert "total_items" in cache_stats
        assert "max_size" in cache_stats


class TestBenchmarkRecord:
    """Test cases for BenchmarkRecord dataclass."""
    
    def test_benchmark_record_creation(self):
        """Test creating a BenchmarkRecord."""
        record = BenchmarkRecord(
            id=1,
            timestamp=time.time(),
            model_name="test_model",
            model_hash="abc123",
            device_info='{"device": "mock_tpu"}',
            benchmark_config='{"iterations": 100}',
            results_json='{"throughput": 100.0}',
            system_info='{"cpu": "test_cpu"}',
            created_at="2024-01-01T00:00:00Z"
        )
        
        assert record.id == 1
        assert record.model_name == "test_model"
        assert record.model_hash == "abc123"
    
    def test_benchmark_record_to_dict(self):
        """Test converting BenchmarkRecord to dictionary."""
        record = BenchmarkRecord(
            id=1,
            timestamp=1234567890.0,
            model_name="test_model",
            model_hash="abc123"
        )
        
        record_dict = record.to_dict()
        
        assert isinstance(record_dict, dict)
        assert record_dict["id"] == 1
        assert record_dict["model_name"] == "test_model"
        assert record_dict["model_hash"] == "abc123"
        assert record_dict["timestamp"] == 1234567890.0


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    def test_full_workflow(self, temp_dir):
        """Test complete database workflow."""
        db_path = temp_dir / "integration_test.db"
        manager = DataManager(db_path=str(db_path))
        
        # Store model metadata
        metadata = {
            "hash_sha256": "integration_hash",
            "name": "integration_model",
            "path": "/path/to/model",
            "format": "onnx",
            "size_bytes": 2000000,
            "input_shape": [1, 3, 224, 224],
            "output_shape": [1, 1000],
            "num_parameters": 1000000,
            "optimization_level": 3,
            "target_device": "tpu_v5_edge",
            "compilation_time_seconds": 10.0,
            "supported_ops_count": 100,
            "unsupported_ops_count": 5
        }
        manager.database.store_model_metadata(metadata)
        
        # Store multiple benchmark results
        for i in range(5):
            manager.store_benchmark_result(
                "integration_model",
                "integration_hash",
                {"device_path": "/dev/apex_0", "tpu_version": "v5_edge"},
                {"iterations": 1000, "warmup": 100, "batch_size": 1},
                {
                    "throughput": 100.0 + i * 5,
                    "latency_metrics": {"p99": 0.01 + i * 0.001, "mean": 0.008 + i * 0.0008},
                    "power_metrics": {"avg_power": 0.85 + i * 0.01, "efficiency": 120 + i * 2}
                },
                {"platform": "Linux", "python_version": "3.8.10"}
            )
        
        # Test data retrieval
        history = manager.get_model_performance_history("integration_model", days=1)
        assert len(history) == 5
        
        # Test performance summary
        summary = manager.get_leaderboard_data(category="all", use_cache=False)
        assert len(summary) >= 1
        
        model_summary = next(s for s in summary if s["model_name"] == "integration_model")
        assert model_summary["benchmark_count"] == 5
        
        # Test system stats
        stats = manager.get_system_stats()
        assert stats["database"]["benchmark_results_count"] == 5
        
        # Test export
        export_path = temp_dir / "export_integration.json"
        manager.export_data(str(export_path), format="json")
        assert export_path.exists()
        
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        assert len(export_data["results"]) == 5
        assert export_data["metadata"]["total_records"] == 5