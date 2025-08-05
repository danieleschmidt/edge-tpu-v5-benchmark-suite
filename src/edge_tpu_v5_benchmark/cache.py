"""Intelligent caching system for TPU v5 benchmark suite."""

from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
import pickle
import json
import hashlib
import logging
import time
import sqlite3
from abc import ABC, abstractmethod
from collections import OrderedDict
import weakref


@dataclass
class CacheEntry:
    """Represents a cached item."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheStorage(ABC):
    """Abstract base class for cache storage backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set cache entry."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all cache keys."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get number of cached items."""
        pass


class MemoryStorage(CacheStorage):
    """In-memory cache storage using LRU eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory_bytes = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry and update LRU order."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired:
                    del self.cache[key]
                    self.current_memory_bytes -= entry.size_bytes
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.touch()
                return entry
            return None
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set cache entry with LRU eviction."""
        with self.lock:
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Check memory limit
            if entry.size_bytes > self.max_memory_bytes:
                return False  # Entry too large
            
            # Evict entries if necessary
            while (len(self.cache) >= self.max_size or 
                   self.current_memory_bytes + entry.size_bytes > self.max_memory_bytes):
                if not self.cache:
                    break
                oldest_key, oldest_entry = self.cache.popitem(last=False)
                self.current_memory_bytes -= oldest_entry.size_bytes
            
            # Add new entry
            self.cache[key] = entry
            self.current_memory_bytes += entry.size_bytes
            return True
    
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.current_memory_bytes -= entry.size_bytes
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_memory_bytes = 0
            return True
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self.lock:
            return list(self.cache.keys())
    
    def size(self) -> int:
        """Get number of cached items."""
        with self.lock:
            return len(self.cache)


class DiskStorage(CacheStorage):
    """Persistent disk-based cache storage."""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.db_path = self.cache_dir / "cache.db"
        self.lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    created_at TEXT,
                    last_accessed TEXT,
                    access_count INTEGER,
                    size_bytes INTEGER,
                    ttl_seconds INTEGER,
                    metadata TEXT,
                    file_path TEXT
                )
            """)
            conn.commit()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry from disk."""
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT * FROM cache_entries WHERE key = ?", (key,)
                    )
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    # Parse row data
                    (db_key, created_at_str, last_accessed_str, access_count, 
                     size_bytes, ttl_seconds, metadata_str, file_path) = row
                    
                    created_at = datetime.fromisoformat(created_at_str)
                    last_accessed = datetime.fromisoformat(last_accessed_str)
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    
                    # Check if expired
                    entry = CacheEntry(
                        key=db_key,
                        value=None,  # Will load from file
                        created_at=created_at,
                        last_accessed=last_accessed,
                        access_count=access_count,
                        size_bytes=size_bytes,
                        ttl_seconds=ttl_seconds,
                        metadata=metadata
                    )
                    
                    if entry.is_expired:
                        self.delete(key)
                        return None
                    
                    # Load value from file
                    cache_file = Path(file_path)
                    if cache_file.exists():
                        with open(cache_file, 'rb') as f:
                            entry.value = pickle.load(f)
                        
                        # Update access statistics
                        entry.touch()
                        conn.execute(
                            "UPDATE cache_entries SET last_accessed = ?, access_count = ? WHERE key = ?",
                            (entry.last_accessed.isoformat(), entry.access_count, key)
                        )
                        conn.commit()
                        
                        return entry
                    
                    return None
                    
            except Exception as e:
                logging.error(f"Error reading from disk cache: {e}")
                return None
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set cache entry to disk."""
        with self.lock:
            try:
                file_path = self._get_file_path(key)
                
                # Write value to file
                with open(file_path, 'wb') as f:
                    pickle.dump(entry.value, f)
                
                # Calculate actual file size
                entry.size_bytes = file_path.stat().st_size
                
                # Check size limit
                if entry.size_bytes > self.max_size_bytes:
                    file_path.unlink()
                    return False
                
                # Store metadata in database
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO cache_entries 
                           (key, created_at, last_accessed, access_count, size_bytes, 
                            ttl_seconds, metadata, file_path)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            key,
                            entry.created_at.isoformat(),
                            entry.last_accessed.isoformat(),
                            entry.access_count,
                            entry.size_bytes,
                            entry.ttl_seconds,
                            json.dumps(entry.metadata),
                            str(file_path)
                        )
                    )
                    conn.commit()
                
                # Cleanup old entries if needed
                self._cleanup_old_entries()
                
                return True
                
            except Exception as e:
                logging.error(f"Error writing to disk cache: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete cache entry from disk."""
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT file_path FROM cache_entries WHERE key = ?", (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        file_path = Path(row[0])
                        if file_path.exists():
                            file_path.unlink()
                        
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
                        return True
                
                return False
                
            except Exception as e:
                logging.error(f"Error deleting from disk cache: {e}")
                return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        with self.lock:
            try:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                
                # Clear database
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("DELETE FROM cache_entries")
                    conn.commit()
                
                return True
                
            except Exception as e:
                logging.error(f"Error clearing disk cache: {e}")
                return False
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute("SELECT key FROM cache_entries")
                    return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logging.error(f"Error reading cache keys: {e}")
                return []
    
    def size(self) -> int:
        """Get number of cached items."""
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                    return cursor.fetchone()[0]
            except Exception as e:
                logging.error(f"Error getting cache size: {e}")
                return 0
    
    def _cleanup_old_entries(self):
        """Cleanup old entries to maintain size limits."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get total cache size
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                total_size = cursor.fetchone()[0] or 0
                
                if total_size > self.max_size_bytes:
                    # Remove oldest entries first
                    cursor = conn.execute(
                        "SELECT key FROM cache_entries ORDER BY last_accessed ASC"
                    )
                    
                    for (key,) in cursor.fetchall():
                        if total_size <= self.max_size_bytes * 0.8:  # Leave 20% headroom
                            break
                        
                        # Get entry size before deletion
                        size_cursor = conn.execute(
                            "SELECT size_bytes FROM cache_entries WHERE key = ?", (key,)
                        )
                        size_row = size_cursor.fetchone()
                        if size_row:
                            total_size -= size_row[0]
                            self.delete(key)
                            
        except Exception as e:
            logging.error(f"Error during cache cleanup: {e}")


class SmartCache:
    """Intelligent caching system with adaptive behavior."""
    
    def __init__(self, 
                 memory_storage: Optional[MemoryStorage] = None,
                 disk_storage: Optional[DiskStorage] = None,
                 default_ttl: int = 3600):  # 1 hour default TTL
        
        self.memory_storage = memory_storage or MemoryStorage()
        self.disk_storage = disk_storage
        self.default_ttl = default_ttl
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'errors': 0
        }
        self.stats_lock = threading.Lock()
        
        # Access pattern tracking
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.pattern_lock = threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with intelligent fallback."""
        # Try memory cache first
        entry = self.memory_storage.get(key)
        if entry:
            self._record_access(key)
            self._update_stats('hits')
            return entry.value
        
        # Try disk cache if available
        if self.disk_storage:
            entry = self.disk_storage.get(key)
            if entry:
                # Promote to memory cache if frequently accessed
                if self._should_promote_to_memory(key, entry):
                    memory_entry = CacheEntry(
                        key=key,
                        value=entry.value,
                        created_at=entry.created_at,
                        last_accessed=datetime.now(),
                        access_count=entry.access_count + 1,
                        size_bytes=self._calculate_size(entry.value),
                        ttl_seconds=entry.ttl_seconds,
                        metadata=entry.metadata
                    )
                    self.memory_storage.set(key, memory_entry)
                
                self._record_access(key)
                self._update_stats('hits')
                return entry.value
        
        self._update_stats('misses')
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            force_disk: bool = False) -> bool:
        """Set value in cache with intelligent storage selection."""
        ttl = ttl or self.default_ttl
        size_bytes = self._calculate_size(value)
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            size_bytes=size_bytes,
            ttl_seconds=ttl,
            metadata={'force_disk': force_disk}
        )
        
        success = False
        
        # Decide storage strategy
        if force_disk or size_bytes > 1024 * 1024:  # > 1MB goes to disk
            if self.disk_storage:
                success = self.disk_storage.set(key, entry)
            else:
                self.logger.warning(f"Large object cached in memory: {key} ({size_bytes} bytes)")
                success = self.memory_storage.set(key, entry)
        else:
            # Try memory first, fallback to disk
            success = self.memory_storage.set(key, entry)
            if not success and self.disk_storage:
                success = self.disk_storage.set(key, entry)
        
        if success:
            self._record_access(key)
            self.logger.debug(f"Cached {key} ({size_bytes} bytes)")
        else:
            self._update_stats('errors')
            self.logger.error(f"Failed to cache {key}")
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache layers."""
        success = False
        
        if self.memory_storage.delete(key):
            success = True
        
        if self.disk_storage and self.disk_storage.delete(key):
            success = True
        
        if success:
            with self.pattern_lock:
                self.access_patterns.pop(key, None)
        
        return success
    
    def clear(self) -> bool:
        """Clear all cache layers."""
        success = True
        
        if not self.memory_storage.clear():
            success = False
        
        if self.disk_storage and not self.disk_storage.clear():
            success = False
        
        with self.pattern_lock:
            self.access_patterns.clear()
        
        with self.stats_lock:
            self.stats = {k: 0 for k in self.stats}
        
        return success
    
    def evict_expired(self) -> int:
        """Evict expired entries from all cache layers."""
        evicted_count = 0
        
        # Check memory cache
        for key in list(self.memory_storage.keys()):
            entry = self.memory_storage.get(key)
            if entry and entry.is_expired:
                self.memory_storage.delete(key)
                evicted_count += 1
        
        # Check disk cache
        if self.disk_storage:
            for key in list(self.disk_storage.keys()):
                entry = self.disk_storage.get(key)
                if entry and entry.is_expired:
                    self.disk_storage.delete(key)
                    evicted_count += 1
        
        if evicted_count > 0:
            self._update_stats('evictions', evicted_count)
            self.logger.info(f"Evicted {evicted_count} expired cache entries")
        
        return evicted_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.stats_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'hit_rate_percent': hit_rate,
                'total_requests': total_requests,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'errors': self.stats['errors'],
                'memory_entries': self.memory_storage.size(),
                'disk_entries': self.disk_storage.size() if self.disk_storage else 0,
                'memory_usage_bytes': getattr(self.memory_storage, 'current_memory_bytes', 0)
            }
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                return 1024  # Default estimate
    
    def _record_access(self, key: str):
        """Record access pattern for intelligent caching decisions."""
        with self.pattern_lock:
            if key not in self.access_patterns:
                self.access_patterns[key] = []
            
            self.access_patterns[key].append(datetime.now())
            
            # Keep only recent access history (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            self.access_patterns[key] = [
                access_time for access_time in self.access_patterns[key]
                if access_time > cutoff
            ]
    
    def _should_promote_to_memory(self, key: str, entry: CacheEntry) -> bool:
        """Decide if disk-cached item should be promoted to memory."""
        with self.pattern_lock:
            access_history = self.access_patterns.get(key, [])
            
            # Promote if accessed frequently (>3 times in last hour)
            recent_accesses = len([
                access_time for access_time in access_history
                if access_time > datetime.now() - timedelta(hours=1)
            ])
            
            return recent_accesses > 3 and entry.size_bytes < 1024 * 1024  # < 1MB
    
    def _update_stats(self, stat: str, increment: int = 1):
        """Update cache statistics."""
        with self.stats_lock:
            self.stats[stat] += increment


# Cache decorators and utilities
def cached(cache: SmartCache, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result
        
        return wrapper
    return decorator


class CacheManager:
    """Global cache manager for the benchmark suite."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".tpu_benchmark_cache"
        self.caches: Dict[str, SmartCache] = {}
        self.logger = logging.getLogger(__name__)
        
        # Create default caches
        self._setup_default_caches()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _setup_default_caches(self):
        """Setup default cache instances."""
        # Model cache - for compiled models
        self.caches['models'] = SmartCache(
            memory_storage=MemoryStorage(max_size=50, max_memory_mb=200),
            disk_storage=DiskStorage(self.cache_dir / "models", max_size_mb=1000),
            default_ttl=86400  # 24 hours
        )
        
        # Results cache - for benchmark results
        self.caches['results'] = SmartCache(
            memory_storage=MemoryStorage(max_size=1000, max_memory_mb=50),
            disk_storage=DiskStorage(self.cache_dir / "results", max_size_mb=500),
            default_ttl=3600  # 1 hour
        )
        
        # Analysis cache - for compiler analysis
        self.caches['analysis'] = SmartCache(
            memory_storage=MemoryStorage(max_size=100, max_memory_mb=50),
            disk_storage=DiskStorage(self.cache_dir / "analysis", max_size_mb=200),
            default_ttl=43200  # 12 hours
        )
        
        # Conversion cache - for model conversions
        self.caches['conversions'] = SmartCache(
            memory_storage=MemoryStorage(max_size=20, max_memory_mb=100),
            disk_storage=DiskStorage(self.cache_dir / "conversions", max_size_mb=2000),
            default_ttl=604800  # 1 week
        )
    
    def get_cache(self, name: str) -> Optional[SmartCache]:
        """Get cache instance by name."""
        return self.caches.get(name)
    
    def create_cache(self, name: str, memory_mb: int = 50, disk_mb: int = 200, 
                    ttl: int = 3600) -> SmartCache:
        """Create a new named cache."""
        cache = SmartCache(
            memory_storage=MemoryStorage(max_size=100, max_memory_mb=memory_mb),
            disk_storage=DiskStorage(self.cache_dir / name, max_size_mb=disk_mb),
            default_ttl=ttl
        )
        self.caches[name] = cache
        return cache
    
    def clear_all(self):
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        self.logger.info("Cleared all caches")
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        total_hit_rate = 0
        total_requests = 0
        
        for name, cache in self.caches.items():
            cache_stats = cache.get_statistics()
            stats[name] = cache_stats
            
            if cache_stats['total_requests'] > 0:
                total_hit_rate += cache_stats['hit_rate_percent'] * cache_stats['total_requests']
                total_requests += cache_stats['total_requests']
        
        global_hit_rate = (total_hit_rate / total_requests) if total_requests > 0 else 0
        
        return {
            'global_hit_rate_percent': global_hit_rate,
            'total_requests': total_requests,
            'cache_count': len(self.caches),
            'caches': stats
        }
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                total_evicted = 0
                for cache in self.caches.values():
                    evicted = cache.evict_expired()
                    total_evicted += evicted
                
                if total_evicted > 0:
                    self.logger.info(f"Cache cleanup evicted {total_evicted} expired entries")
                    
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_cache(name: str) -> Optional[SmartCache]:
    """Get named cache instance."""
    return get_cache_manager().get_cache(name)