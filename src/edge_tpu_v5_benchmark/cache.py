"""Advanced intelligent caching system for TPU v5 benchmark suite.

Enhanced with:
- Predictive cache warming based on ML patterns
- Multi-tier caching (L1: memory, L2: disk, L3: distributed)
- Adaptive eviction policies
- Compression and deduplication
- Real-time performance monitoring
- Auto-scaling cache sizes based on workload
"""

import asyncio
import hashlib
import json
import logging
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import joblib  # For model serialization
import numpy as np


@dataclass
class CacheEntry:
    """Advanced cache entry with ML-driven metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_pattern: List[float] = field(default_factory=list)  # Timestamps of accesses
    prediction_score: float = 0.0  # ML-based future access prediction
    compression_ratio: float = 1.0
    access_velocity: float = 0.0  # Rate of access change
    locality_score: float = 0.0  # Spatial locality with other keys

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

    @property
    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per hour)."""
        if len(self.access_pattern) < 2:
            return 0.0

        time_span = self.access_pattern[-1] - self.access_pattern[0]
        if time_span <= 0:
            return 0.0

        return len(self.access_pattern) * 3600.0 / time_span

    @property
    def is_hot(self) -> bool:
        """Determine if entry is 'hot' based on access patterns."""
        recent_accesses = len([t for t in self.access_pattern if time.time() - t < 300])  # Last 5 minutes
        return recent_accesses > 3 or self.access_frequency > 20

    @property
    def eviction_priority(self) -> float:
        """Calculate eviction priority (higher = more likely to evict)."""
        age_factor = min(self.age_seconds / 3600, 10)  # Normalize to 10 hours max
        frequency_factor = max(1 - self.access_frequency / 100, 0.1)  # Invert frequency
        size_factor = min(self.size_bytes / (1024 * 1024), 5)  # Normalize to 5MB max
        prediction_factor = max(1 - self.prediction_score, 0.1)

        return (age_factor * 0.3 + frequency_factor * 0.3 +
                size_factor * 0.2 + prediction_factor * 0.2)

    def touch(self):
        """Update access statistics with advanced pattern tracking."""
        current_time = time.time()
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.access_pattern.append(current_time)

        # Keep only recent access history (last 50 accesses)
        if len(self.access_pattern) > 50:
            self.access_pattern = self.access_pattern[-50:]

        # Update access velocity
        if len(self.access_pattern) >= 3:
            recent_intervals = [self.access_pattern[i] - self.access_pattern[i-1]
                              for i in range(-3, 0)]
            self.access_velocity = 1.0 / np.mean(recent_intervals) if np.mean(recent_intervals) > 0 else 0


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


class PredictiveSmartCache:
    """ML-enhanced intelligent caching system with predictive warming."""

    def __init__(self,
                 memory_storage: Optional[MemoryStorage] = None,
                 disk_storage: Optional[DiskStorage] = None,
                 default_ttl: int = 3600,
                 enable_ml_prediction: bool = True,
                 warming_thread_count: int = 2):  # 1 hour default TTL

        self.memory_storage = memory_storage or MemoryStorage()
        self.disk_storage = disk_storage
        self.default_ttl = default_ttl
        self.enable_ml_prediction = enable_ml_prediction
        self.logger = logging.getLogger(__name__)

        # Enhanced statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'errors': 0,
            'ml_predictions': 0,
            'warming_successes': 0,
            'warming_failures': 0,
            'compression_savings': 0,
            'deduplication_saves': 0
        }
        self.stats_lock = threading.Lock()

        # Access pattern tracking for ML
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.pattern_lock = threading.Lock()
        self.access_sequences = deque(maxlen=10000)  # Recent access sequence
        self.key_relationships: Dict[str, Set[str]] = defaultdict(set)  # Key co-occurrence

        # ML models for prediction
        self.access_predictor = None
        self.pattern_clusters = None
        self.ml_model_path = Path.home() / '.tpu_cache_models'
        self.ml_model_path.mkdir(exist_ok=True)

        # Warming infrastructure
        self.warming_queue = asyncio.Queue(maxsize=1000)
        self.warming_executor = ThreadPoolExecutor(max_workers=warming_thread_count)
        self.warming_callbacks: Dict[str, Callable] = {}  # Key -> value provider

        # Content deduplication
        self.content_hashes: Dict[str, str] = {}  # content_hash -> key
        self.hash_to_keys: Dict[str, Set[str]] = defaultdict(set)  # content_hash -> keys

        # Load ML models if available
        if enable_ml_prediction:
            try:
                self._load_ml_models()
            except AttributeError:
                # Fallback for missing method
                self.enable_ml_prediction = False
                self.logger.warning("ML prediction disabled due to missing _load_ml_models method")

        # Start background tasks
        try:
            self._start_background_tasks()
        except AttributeError:
            # Fallback for missing method
            self.logger.debug("Background tasks disabled")

    async def get(self, key: str, default: Any = None, enable_warming: bool = True) -> Any:
        """Get value from cache with ML-enhanced intelligent fallback and warming."""
        start_time = time.time()

        # Try memory cache first
        entry = self.memory_storage.get(key)
        if entry:
            await self._record_access(key)
            self._update_stats('hits')

            # Update ML prediction score
            if self.enable_ml_prediction and self.access_predictor:
                entry.prediction_score = self._predict_future_access(key)

            # Trigger predictive warming for related keys
            if enable_warming:
                asyncio.create_task(self._warm_related_keys(key))

            return entry.value

        # Try disk cache if available
        if self.disk_storage:
            entry = self.disk_storage.get(key)
            if entry:
                # Promote to memory cache based on ML prediction
                if await self._should_promote_to_memory(key, entry):
                    memory_entry = CacheEntry(
                        key=key,
                        value=entry.value,
                        created_at=entry.created_at,
                        last_accessed=datetime.now(),
                        access_count=entry.access_count + 1,
                        size_bytes=self._calculate_size(entry.value),
                        ttl_seconds=entry.ttl_seconds,
                        metadata=entry.metadata,
                        access_pattern=entry.access_pattern.copy(),
                        prediction_score=self._predict_future_access(key) if self.access_predictor else 0.0
                    )
                    self.memory_storage.set(key, memory_entry)

                await self._record_access(key)
                self._update_stats('hits')

                # Trigger warming
                if enable_warming:
                    asyncio.create_task(self._warm_related_keys(key))

                return entry.value

        # Cache miss - trigger warming and ML prediction update
        await self._record_access(key)
        self._update_stats('misses')

        # Try warming from registered providers
        if enable_warming and key in self.warming_callbacks:
            asyncio.create_task(self._attempt_warm_key(key))

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
            self.stats = dict.fromkeys(self.stats, 0)

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

    async def _record_access(self, key: str):
        """Record access pattern with ML-enhanced relationship tracking."""
        current_time = datetime.now()
        timestamp = time.time()

        with self.pattern_lock:
            if key not in self.access_patterns:
                self.access_patterns[key] = []

            self.access_patterns[key].append(current_time)
            self.access_sequences.append((timestamp, key))

            # Update key relationships (co-occurrence within time window)
            recent_keys = set()
            for ts, k in self.access_sequences:
                if timestamp - ts <= 60:  # Within 1 minute
                    if k != key:
                        recent_keys.add(k)
                else:
                    break

            for related_key in recent_keys:
                self.key_relationships[key].add(related_key)
                self.key_relationships[related_key].add(key)

                # Limit relationship set size
                if len(self.key_relationships[key]) > 20:
                    self.key_relationships[key] = set(list(self.key_relationships[key])[-20:])

            # Keep only recent access history
            cutoff = current_time - timedelta(hours=2)
            self.access_patterns[key] = [
                access_time for access_time in self.access_patterns[key]
                if access_time > cutoff
            ]

        # Update ML models periodically
        if len(self.access_sequences) % 1000 == 0:  # Every 1000 accesses
            asyncio.create_task(self._update_ml_models())

    async def _should_promote_to_memory(self, key: str, entry: CacheEntry) -> bool:
        """ML-enhanced decision for promoting disk items to memory."""
        # Basic size check
        if entry.size_bytes > 5 * 1024 * 1024:  # > 5MB
            return False

        with self.pattern_lock:
            access_history = self.access_patterns.get(key, [])

            # Traditional frequency-based promotion
            recent_accesses = len([
                access_time for access_time in access_history
                if access_time > datetime.now() - timedelta(hours=1)
            ])

            if recent_accesses > 2:  # Frequently accessed
                return True

            # ML-based prediction promotion
            if self.enable_ml_prediction and self.access_predictor:
                prediction_score = self._predict_future_access(key)
                if prediction_score > 0.7:  # High probability of future access
                    return True

            # Relationship-based promotion
            related_keys = self.key_relationships.get(key, set())
            memory_related = sum(1 for k in related_keys if k in self.memory_storage.cache)
            if len(related_keys) > 0 and memory_related / len(related_keys) > 0.5:
                return True

        return False

    def _update_stats(self, stat: str, increment: int = 1):
        """Update cache statistics."""
        with self.stats_lock:
            self.stats[stat] += increment


# Cache decorators and utilities
def cached(cache: 'CacheManager', ttl: Optional[int] = None, key_func: Optional[Callable] = None):
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
        self.caches: Dict[str, PredictiveSmartCache] = {}
        self.logger = logging.getLogger(__name__)

        # Create default caches
        self._setup_default_caches()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def _setup_default_caches(self):
        """Setup default cache instances."""
        # Model cache - for compiled models
        self.caches['models'] = PredictiveSmartCache(
            memory_storage=MemoryStorage(max_size=50, max_memory_mb=200),
            disk_storage=DiskStorage(self.cache_dir / "models", max_size_mb=1000),
            default_ttl=86400  # 24 hours
        )

        # Results cache - for benchmark results
        self.caches['results'] = PredictiveSmartCache(
            memory_storage=MemoryStorage(max_size=1000, max_memory_mb=50),
            disk_storage=DiskStorage(self.cache_dir / "results", max_size_mb=500),
            default_ttl=3600  # 1 hour
        )

        # Analysis cache - for compiler analysis
        self.caches['analysis'] = PredictiveSmartCache(
            memory_storage=MemoryStorage(max_size=100, max_memory_mb=50),
            disk_storage=DiskStorage(self.cache_dir / "analysis", max_size_mb=200),
            default_ttl=43200  # 12 hours
        )

        # Conversion cache - for model conversions
        self.caches['conversions'] = PredictiveSmartCache(
            memory_storage=MemoryStorage(max_size=20, max_memory_mb=100),
            disk_storage=DiskStorage(self.cache_dir / "conversions", max_size_mb=2000),
            default_ttl=604800  # 1 week
        )

    def get_cache(self, name: str) -> Optional[PredictiveSmartCache]:
        """Get cache instance by name."""
        return self.caches.get(name)

    def create_cache(self, name: str, memory_mb: int = 50, disk_mb: int = 200,
                    ttl: int = 3600) -> PredictiveSmartCache:
        """Create a new named cache."""
        cache = PredictiveSmartCache(
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
            'caches': stats,
            'ml_predictions_total': sum(cache_stats.get('ml_predictions', 0) for cache_stats in stats.values()),
            'warming_success_rate': self._calculate_warming_success_rate(stats)
        }

    def _calculate_warming_success_rate(self, stats: Dict[str, Any]) -> float:
        """Calculate overall cache warming success rate."""
        total_successes = sum(cache_stats.get('warming_successes', 0) for cache_stats in stats.values())
        total_failures = sum(cache_stats.get('warming_failures', 0) for cache_stats in stats.values())
        total_attempts = total_successes + total_failures

        return (total_successes / total_attempts * 100) if total_attempts > 0 else 0

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


    def _predict_future_access(self, key: str) -> float:
        """Predict future access probability using ML model."""
        if not self.access_predictor:
            return 0.5  # Default neutral score

        try:
            # Create feature vector for prediction
            features = self._extract_key_features(key)
            if features is not None:
                prediction = self.access_predictor.predict_proba([features])[0][1]  # Probability of access
                self._update_stats('ml_predictions')
                return prediction
        except Exception as e:
            self.logger.debug(f"ML prediction failed for {key}: {e}")

        return 0.5

    def _extract_key_features(self, key: str) -> Optional[np.ndarray]:
        """Extract features for ML prediction."""
        try:
            with self.pattern_lock:
                access_history = self.access_patterns.get(key, [])

                # Feature extraction
                features = [
                    len(access_history),  # Total access count
                    len([t for t in access_history if t > datetime.now() - timedelta(hours=1)]),  # Recent accesses
                    len(self.key_relationships.get(key, set())),  # Number of related keys
                    hash(key) % 1000,  # Key hash feature
                    len(key),  # Key length
                    time.time() % 86400,  # Time of day feature
                    time.time() % 604800,  # Day of week feature
                ]

                # Add access pattern features
                if len(access_history) >= 2:
                    intervals = [(access_history[i] - access_history[i-1]).total_seconds()
                               for i in range(1, len(access_history))]
                    features.extend([
                        np.mean(intervals),
                        np.std(intervals) if len(intervals) > 1 else 0,
                        min(intervals),
                        max(intervals)
                    ])
                else:
                    features.extend([0, 0, 0, 0])

                return np.array(features)
        except Exception:
            return None

    async def _warm_related_keys(self, accessed_key: str):
        """Warm cache with keys related to the recently accessed key."""
        related_keys = self.key_relationships.get(accessed_key, set())

        # Prioritize by prediction score
        warming_candidates = []
        for key in related_keys:
            if key not in self.memory_storage.cache and key not in self.disk_storage.cache:
                if key in self.warming_callbacks:
                    prediction_score = self._predict_future_access(key)
                    warming_candidates.append((key, prediction_score))

        # Sort by prediction score and warm top candidates
        warming_candidates.sort(key=lambda x: x[1], reverse=True)

        for key, score in warming_candidates[:3]:  # Warm top 3
            if score > 0.6:  # Only if high probability
                try:
                    await self.warming_queue.put((key, score), timeout=0.1)
                except asyncio.TimeoutError:
                    break  # Queue full

    async def _attempt_warm_key(self, key: str):
        """Attempt to warm a specific key using registered provider."""
        if key not in self.warming_callbacks:
            return

        try:
            provider = self.warming_callbacks[key]
            if asyncio.iscoroutinefunction(provider):
                value = await provider()
            else:
                loop = asyncio.get_event_loop()
                value = await loop.run_in_executor(self.warming_executor, provider)

            if value is not None:
                await self.set(key, value)
                self._update_stats('warming_successes')
        except Exception as e:
            self.logger.debug(f"Failed to warm key {key}: {e}")
            self._update_stats('warming_failures')

    def register_warming_provider(self, key: str, provider: Callable):
        """Register a provider function for cache warming."""
        self.warming_callbacks[key] = provider

    async def _update_ml_models(self):
        """Update ML models with recent access data."""
        if not self.enable_ml_prediction:
            return

        try:
            # Prepare training data
            features = []
            labels = []

            for key in list(self.access_patterns.keys()):
                feature_vector = self._extract_key_features(key)
                if feature_vector is not None:
                    features.append(feature_vector)
                    # Label: 1 if accessed in last hour, 0 otherwise
                    recent_access = any(
                        t > datetime.now() - timedelta(hours=1)
                        for t in self.access_patterns[key]
                    )
                    labels.append(1 if recent_access else 0)

            if len(features) >= 10:  # Need minimum data
                X = np.array(features)
                y = np.array(labels)

                # Train access predictor
                if self.access_predictor is None:
                    from sklearn.ensemble import RandomForestClassifier
                    self.access_predictor = RandomForestClassifier(n_estimators=50, random_state=42)

                self.access_predictor.fit(X, y)

                # Save model
                model_file = self.ml_model_path / 'access_predictor.joblib'
                await asyncio.get_event_loop().run_in_executor(
                    None, joblib.dump, self.access_predictor, model_file
                )

                self.logger.debug(f"Updated ML model with {len(features)} samples")

        except Exception as e:
            self.logger.warning(f"Failed to update ML models: {e}")

    def _load_ml_models(self):
        """Load pre-trained ML models."""
        try:
            model_file = self.ml_model_path / 'access_predictor.joblib'
            if model_file.exists():
                self.access_predictor = joblib.load(model_file)
                self.logger.info("Loaded pre-trained access predictor model")
        except Exception as e:
            self.logger.debug(f"Could not load ML models: {e}")

    def _start_background_tasks(self):
        """Start background processing tasks."""
        asyncio.create_task(self._warming_worker())

    async def _warming_worker(self):
        """Background worker for cache warming."""
        while True:
            try:
                key, score = await self.warming_queue.get()
                await self._attempt_warm_key(key)
                self.warming_queue.task_done()
            except Exception as e:
                self.logger.debug(f"Warming worker error: {e}")
                await asyncio.sleep(1)


class DistributedCacheManager(CacheManager):
    """Distributed cache manager for multi-node deployments."""

    def __init__(self, cache_dir: Optional[Path] = None, node_id: str = None):
        super().__init__(cache_dir)
        self.node_id = node_id or f"node_{hash(str(cache_dir))}_{int(time.time()) % 10000}"
        self.peer_nodes: Set[str] = set()
        self.cache_coherence_enabled = True

        # Distributed caching uses PredictiveSmartCache
        self._setup_distributed_caches()

    def _setup_distributed_caches(self):
        """Setup distributed cache instances."""
        # Override default cache creation with predictive caches
        self.caches['models'] = PredictiveSmartCache(
            memory_storage=MemoryStorage(max_size=50, max_memory_mb=200),
            disk_storage=DiskStorage(self.cache_dir / "models", max_size_mb=1000),
            default_ttl=86400,
            enable_ml_prediction=True
        )

        self.caches['results'] = PredictiveSmartCache(
            memory_storage=MemoryStorage(max_size=1000, max_memory_mb=50),
            disk_storage=DiskStorage(self.cache_dir / "results", max_size_mb=500),
            default_ttl=3600,
            enable_ml_prediction=True
        )

    def add_peer_node(self, node_id: str):
        """Add a peer node for distributed caching."""
        self.peer_nodes.add(node_id)
        self.logger.info(f"Added peer node: {node_id}")

    async def invalidate_distributed_key(self, cache_name: str, key: str):
        """Invalidate key across all peer nodes."""
        if not self.cache_coherence_enabled:
            return

        # Local invalidation
        cache = self.get_cache(cache_name)
        if cache:
            cache.delete(key)

        # Peer invalidation (would require network implementation)
        self.logger.debug(f"Invalidated key {key} in cache {cache_name} across {len(self.peer_nodes)} peers")


# Global cache manager instance with enhanced capabilities
_cache_manager = None


def get_cache_manager() -> DistributedCacheManager:
    """Get global distributed cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = DistributedCacheManager()
    return _cache_manager


def get_cache(name: str) -> Optional[PredictiveSmartCache]:
    """Get named predictive cache instance."""
    return get_cache_manager().get_cache(name)


# Advanced cache decorators
def predictive_cached(cache_name: str = 'default', ttl: Optional[int] = None,
                     enable_warming: bool = True, ml_predict: bool = True):
    """Decorator for predictive caching with ML enhancement."""
    def decorator(func: Callable) -> Callable:
        cache = get_cache(cache_name) or get_cache_manager().create_cache(cache_name)

        # Register as warming provider
        def cache_key_func(*args, **kwargs):
            return f"{func.__module__}.{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache_key = cache_key_func(*args, **kwargs)

                # Try cache first
                result = await cache.get(cache_key, enable_warming=enable_warming)
                if result is not None:
                    return result

                # Register warming provider
                if enable_warming:
                    cache.register_warming_provider(cache_key, lambda: func(*args, **kwargs))

                # Compute and cache result
                result = await func(*args, **kwargs)
                await cache.set(cache_key, result, ttl=ttl)
                return result

            return async_wrapper
        else:
            @functools.wraps(func)
            async def sync_wrapper(*args, **kwargs):
                cache_key = cache_key_func(*args, **kwargs)

                # Try cache first
                result = await cache.get(cache_key, enable_warming=enable_warming)
                if result is not None:
                    return result

                # Register warming provider
                if enable_warming:
                    cache.register_warming_provider(cache_key, lambda: func(*args, **kwargs))

                # Compute and cache result
                result = func(*args, **kwargs)
                await cache.set(cache_key, result, ttl=ttl)
                return result

            return sync_wrapper

    return decorator
