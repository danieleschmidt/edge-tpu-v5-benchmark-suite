"""Unit tests for caching system."""

import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from edge_tpu_v5_benchmark.cache import (
    CacheEntry, MemoryStorage, DiskStorage, PredictiveSmartCache, 
    CacheManager, cached
)


class TestCacheEntry:
    """Test CacheEntry functionality."""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=3600
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert not entry.is_expired
        assert entry.age_seconds >= 0
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        # Create expired entry
        old_time = datetime.now() - timedelta(seconds=7200)  # 2 hours ago
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=old_time,
            last_accessed=old_time,
            ttl_seconds=3600  # 1 hour TTL
        )
        
        assert entry.is_expired
        assert entry.age_seconds > 7000  # Should be around 7200 seconds
    
    def test_cache_entry_touch(self):
        """Test cache entry touch functionality."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0
        )
        
        initial_access_count = entry.access_count
        entry.touch()
        
        assert entry.access_count == initial_access_count + 1
        assert entry.last_accessed > entry.created_at


class TestMemoryStorage:
    """Test MemoryStorage functionality."""
    
    def test_memory_storage_basic_operations(self):
        """Test basic get/set/delete operations."""
        storage = MemoryStorage(max_size=10)
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            size_bytes=100
        )
        
        # Test set and get
        assert storage.set("test_key", entry)
        retrieved = storage.get("test_key")
        assert retrieved is not None
        assert retrieved.value == "test_value"
        
        # Test delete
        assert storage.delete("test_key")
        assert storage.get("test_key") is None
        
        # Test delete non-existent key
        assert not storage.delete("non_existent")
    
    def test_memory_storage_lru_eviction(self):
        """Test LRU eviction in memory storage."""
        storage = MemoryStorage(max_size=2)
        
        # Add two entries
        entry1 = CacheEntry("key1", "value1", datetime.now(), datetime.now(), size_bytes=50)
        entry2 = CacheEntry("key2", "value2", datetime.now(), datetime.now(), size_bytes=50)
        
        storage.set("key1", entry1)
        storage.set("key2", entry2)
        
        # Access key1 to make it most recently used
        storage.get("key1")
        
        # Add third entry, should evict key2
        entry3 = CacheEntry("key3", "value3", datetime.now(), datetime.now(), size_bytes=50)
        storage.set("key3", entry3)
        
        assert storage.get("key1") is not None  # Still present
        assert storage.get("key2") is None      # Evicted
        assert storage.get("key3") is not None  # Newly added
    
    def test_memory_storage_size_limit(self):
        """Test memory storage size limits."""
        storage = MemoryStorage(max_size=10, max_memory_mb=1)  # 1MB limit
        
        # Try to store an entry larger than limit
        large_entry = CacheEntry(
            "large_key", 
            "x" * (2 * 1024 * 1024),  # 2MB
            datetime.now(), 
            datetime.now(), 
            size_bytes=2 * 1024 * 1024
        )
        
        assert not storage.set("large_key", large_entry)
        assert storage.get("large_key") is None
    
    def test_memory_storage_expired_entries(self):
        """Test handling of expired entries."""
        storage = MemoryStorage()
        
        # Create expired entry
        old_time = datetime.now() - timedelta(seconds=7200)
        expired_entry = CacheEntry(
            "expired_key",
            "expired_value",
            old_time,
            old_time,
            ttl_seconds=3600,
            size_bytes=50
        )
        
        storage.set("expired_key", expired_entry)
        
        # Should return None for expired entry
        assert storage.get("expired_key") is None
        
        # Entry should be automatically removed
        assert "expired_key" not in storage.cache


class TestDiskStorage:
    """Test DiskStorage functionality."""
    
    def test_disk_storage_basic_operations(self):
        """Test basic disk storage operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorage(Path(tmpdir), max_size_mb=10)
            
            entry = CacheEntry(
                "test_key",
                {"data": "test_value"},
                datetime.now(),
                datetime.now(),
                size_bytes=100
            )
            
            # Test set and get
            assert storage.set("test_key", entry)
            retrieved = storage.get("test_key")
            assert retrieved is not None
            assert retrieved.value["data"] == "test_value"
            
            # Test keys and size
            assert "test_key" in storage.keys()
            assert storage.size() == 1
            
            # Test delete
            assert storage.delete("test_key")
            assert storage.get("test_key") is None
            assert storage.size() == 0
    
    def test_disk_storage_persistence(self):
        """Test disk storage persistence across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create first storage instance and add entry
            storage1 = DiskStorage(cache_dir)
            entry = CacheEntry(
                "persistent_key",
                "persistent_value",
                datetime.now(),
                datetime.now(),
                size_bytes=100
            )
            storage1.set("persistent_key", entry)
            
            # Create second storage instance
            storage2 = DiskStorage(cache_dir)
            
            # Should be able to retrieve entry from second instance
            retrieved = storage2.get("persistent_key")
            assert retrieved is not None
            assert retrieved.value == "persistent_value"
    
    def test_disk_storage_clear(self):
        """Test disk storage clear operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorage(Path(tmpdir))
            
            # Add multiple entries
            for i in range(5):
                entry = CacheEntry(
                    f"key_{i}",
                    f"value_{i}",
                    datetime.now(),
                    datetime.now(),
                    size_bytes=100
                )
                storage.set(f"key_{i}", entry)
            
            assert storage.size() == 5
            
            # Clear all entries
            assert storage.clear()
            assert storage.size() == 0
            assert len(storage.keys()) == 0


class TestPredictiveSmartCache:
    """Test SmartCache functionality."""
    
    def test_smart_cache_basic_operations(self):
        """Test basic smart cache operations."""
        cache = PredictiveSmartCache()
        
        # Test set and get
        assert cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        # Test default value
        assert cache.get("non_existent", "default") == "default"
        
        # Test delete
        assert cache.delete("test_key")
        assert cache.get("test_key") is None
    
    def test_smart_cache_statistics(self):
        """Test smart cache statistics."""
        cache = PredictiveSmartCache()
        
        # Initially no stats
        stats = cache.get_statistics()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Generate some hits and misses
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_statistics()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_percent"] == 50.0
    
    def test_smart_cache_ttl(self):
        """Test smart cache TTL functionality."""
        cache = PredictiveSmartCache(default_ttl=1)  # 1 second TTL
        
        cache.set("ttl_key", "ttl_value")
        assert cache.get("ttl_key") == "ttl_value"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should expire
        expired_count = cache.evict_expired()
        assert expired_count > 0
        assert cache.get("ttl_key") is None
    
    def test_smart_cache_large_objects(self):
        """Test smart cache handling of large objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            disk_storage = DiskStorage(Path(tmpdir))
            cache = PredictiveSmartCache(disk_storage=disk_storage)
            
            # Large object should go to disk
            large_value = "x" * (2 * 1024 * 1024)  # 2MB
            cache.set("large_key", large_value, force_disk=True)
            
            retrieved = cache.get("large_key")
            assert retrieved == large_value


class TestCacheDecorators:
    """Test cache decorators."""
    
    def test_cached_decorator(self):
        """Test cached function decorator."""
        cache = PredictiveSmartCache()
        call_count = 0
        
        @cached(cache, ttl=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again
        
        # Different arguments should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_cached_decorator_with_key_func(self):
        """Test cached decorator with custom key function."""
        cache = PredictiveSmartCache()
        
        def custom_key(*args, **kwargs):
            return f"custom:{args[0]}"
        
        @cached(cache, key_func=custom_key)
        def test_function(x):
            return x * 2
        
        result1 = test_function(5)
        assert result1 == 10
        
        # Should use same cache key
        result2 = test_function(5)
        assert result2 == 10


class TestCacheManager:
    """Test CacheManager functionality."""
    
    def test_cache_manager_default_caches(self):
        """Test cache manager default caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))
            
            # Should have default caches
            assert manager.get_cache("models") is not None
            assert manager.get_cache("results") is not None
            assert manager.get_cache("analysis") is not None
            assert manager.get_cache("conversions") is not None
    
    def test_cache_manager_custom_cache(self):
        """Test creating custom caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))
            
            # Create custom cache
            custom_cache = manager.create_cache("custom", memory_mb=25, disk_mb=100, ttl=1800)
            assert custom_cache is not None
            
            # Should be retrievable
            assert manager.get_cache("custom") is custom_cache
    
    def test_cache_manager_statistics(self):
        """Test cache manager global statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))
            
            # Use some caches
            models_cache = manager.get_cache("models")
            models_cache.set("test_model", "model_data")
            models_cache.get("test_model")  # Hit
            models_cache.get("non_existent")  # Miss
            
            stats = manager.get_global_statistics()
            assert stats["total_requests"] > 0
            assert stats["cache_count"] > 0
            assert "models" in stats["caches"]
    
    def test_cache_manager_clear_all(self):
        """Test clearing all caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))
            
            # Add data to multiple caches
            manager.get_cache("models").set("key1", "value1")
            manager.get_cache("results").set("key2", "value2")
            
            # Clear all
            manager.clear_all()
            
            # All should be empty
            assert manager.get_cache("models").get("key1") is None
            assert manager.get_cache("results").get("key2") is None


# Integration tests
class TestCacheIntegration:
    """Integration tests for cache system."""
    
    def test_memory_disk_promotion(self):
        """Test promotion from disk to memory cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PredictiveSmartCache(disk_storage=DiskStorage(Path(tmpdir)))
            
            # Set value that goes to disk
            cache.set("promote_key", "promote_value", force_disk=True)
            
            # Access multiple times to trigger promotion
            for _ in range(5):
                cache.get("promote_key")
            
            # Should now be in memory cache
            memory_entry = cache.memory_storage.get("promote_key")
            assert memory_entry is not None
    
    def test_concurrent_cache_access(self):
        """Test concurrent cache access."""
        import threading
        
        cache = PredictiveSmartCache()
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(100):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    
                    cache.set(key, value)
                    retrieved = cache.get(key)
                    
                    if retrieved != value:
                        errors.append(f"Mismatch for {key}: expected {value}, got {retrieved}")
                    else:
                        results.append(key)
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 500  # 5 workers * 100 operations each


if __name__ == "__main__":
    pytest.main([__file__])