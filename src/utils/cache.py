"""
src/utils/cache.py — Disk-based caching with TTL and decorator support.

Cached values are stored as pickle files named by the SHA-256 hash of their
string key.  Each file wraps the actual value in a metadata envelope so that
TTL expiry can be detected without loading the full payload.

Usage
-----
    from src.utils.cache import get_cache, cached

    weather_cache = get_cache("weather", ttl_seconds=1800)

    # Low-level API
    weather_cache.set("mumbai_forecast", data)
    data = weather_cache.get("mumbai_forecast")  # None if expired

    # Decorator API
    @cached(weather_cache)
    def fetch_forecast(city: str) -> dict: ...
"""

from __future__ import annotations

import functools
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# DiskCache
# ---------------------------------------------------------------------------

class DiskCache:
    """
    A simple, file-based key-value cache with TTL support.

    Each cache entry is stored as a pickle file whose name is the SHA-256
    digest of the string key.  The pickled payload is a dict::

        {
            "value":      <any picklable object>,
            "stored_at":  <float — Unix timestamp>,
            "ttl":        <int — seconds>
        }

    Parameters
    ----------
    cache_dir:
        Directory to store cache files in.  Created on first use.
    ttl_seconds:
        Default time-to-live in seconds.  Entries older than this are
        treated as cache misses.  Pass ``0`` to disable expiry.
    """

    def __init__(self, cache_dir: str, ttl_seconds: int = 3600) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, key: str) -> Path:
        """
        Return the filesystem path for *key*.

        The filename is the first 64 hex characters of the SHA-256 digest
        of the UTF-8 encoded key string.

        Parameters
        ----------
        key:
            Arbitrary string key.

        Returns
        -------
        Path
            Absolute path to the corresponding ``.pkl`` cache file.
        """
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.pkl"

    def _is_expired(self, envelope: dict) -> bool:
        """Return ``True`` if the cache entry has exceeded its TTL."""
        ttl = envelope.get("ttl", self.ttl_seconds)
        if ttl == 0:
            return False  # TTL of 0 means never expire
        age = time.time() - envelope.get("stored_at", 0.0)
        return age > ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value by *key*.

        Returns ``None`` if the key does not exist or the entry has
        exceeded its TTL.  The expired file is **not** deleted on a miss
        — call :meth:`clear_expired` to clean up.

        Parameters
        ----------
        key:
            Cache key to look up.

        Returns
        -------
        Optional[Any]
            The cached value, or ``None`` on a cache miss / expiry.
        """
        path = self._make_key(key)
        if not path.exists():
            return None

        try:
            with path.open("rb") as fh:
                envelope: dict = pickle.load(fh)
        except (OSError, pickle.UnpicklingError, EOFError):
            # Corrupt or unreadable file — treat as a miss
            return None

        if self._is_expired(envelope):
            return None

        return envelope.get("value")

    def set(self, key: str, value: Any) -> None:
        """
        Store *value* under *key*.

        The entry will expire after ``self.ttl_seconds`` seconds from now
        (or never, if ``ttl_seconds`` is ``0``).

        Parameters
        ----------
        key:
            Cache key.
        value:
            Any picklable Python object.
        """
        envelope: dict = {
            "value": value,
            "stored_at": time.time(),
            "ttl": self.ttl_seconds,
        }
        path = self._make_key(key)
        try:
            with path.open("wb") as fh:
                pickle.dump(envelope, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError as exc:
            # Non-fatal: a failed cache write should not abort the caller
            import warnings
            warnings.warn(
                f"DiskCache.set failed for key {key!r}: {exc}", RuntimeWarning, stacklevel=2
            )

    def invalidate(self, key: str) -> bool:
        """
        Remove the cache entry for *key* if it exists.

        Parameters
        ----------
        key:
            Cache key to invalidate.

        Returns
        -------
        bool
            ``True`` if the file existed and was deleted, ``False`` otherwise.
        """
        path = self._make_key(key)
        if path.exists():
            path.unlink(missing_ok=True)
            return True
        return False

    def clear_expired(self) -> int:
        """
        Scan the cache directory and delete all expired entries.

        Returns
        -------
        int
            Number of files removed.
        """
        removed = 0
        for pkl_file in self._dir.glob("*.pkl"):
            try:
                with pkl_file.open("rb") as fh:
                    envelope: dict = pickle.load(fh)
                if self._is_expired(envelope):
                    pkl_file.unlink(missing_ok=True)
                    removed += 1
            except (OSError, pickle.UnpicklingError, EOFError):
                # Unreadable file — remove it too
                pkl_file.unlink(missing_ok=True)
                removed += 1
        return removed

    def cache_info(self) -> dict:
        """
        Return summary statistics about the cache directory.

        Returns
        -------
        dict
            A dictionary with keys:

            ``total_entries`` : int
                Total number of ``.pkl`` files in the cache directory.
            ``expired`` : int
                Number of entries that have exceeded their TTL.
            ``valid`` : int
                Number of entries that are still within TTL.
            ``size_mb`` : float
                Total disk usage of all cache files in megabytes, rounded
                to two decimal places.
        """
        total = 0
        expired = 0
        total_bytes = 0

        for pkl_file in self._dir.glob("*.pkl"):
            total += 1
            total_bytes += pkl_file.stat().st_size
            try:
                with pkl_file.open("rb") as fh:
                    envelope: dict = pickle.load(fh)
                if self._is_expired(envelope):
                    expired += 1
            except (OSError, pickle.UnpicklingError, EOFError):
                expired += 1  # Treat corrupt files as expired

        return {
            "total_entries": total,
            "expired": expired,
            "valid": total - expired,
            "size_mb": round(total_bytes / (1024 * 1024), 2),
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_cache_registry: dict[str, DiskCache] = {}


def get_cache(name: str, ttl_seconds: int = 3600) -> DiskCache:
    """
    Return (or create) a named :class:`DiskCache` instance.

    Named caches are singletons — calling ``get_cache("weather")`` twice
    returns the same object.  The cache directory is ``./cache/<name>/``.

    Parameters
    ----------
    name:
        Logical cache name (used as the subdirectory under ``./cache/``).
    ttl_seconds:
        Default TTL for entries in this cache.

    Returns
    -------
    DiskCache
        The named cache instance.
    """
    if name not in _cache_registry:
        _cache_registry[name] = DiskCache(
            cache_dir=f"./cache/{name}", ttl_seconds=ttl_seconds
        )
    return _cache_registry[name]


# ---------------------------------------------------------------------------
# Caching decorator
# ---------------------------------------------------------------------------

def cached(
    cache: DiskCache,
    key_fn: Optional[Callable[..., str]] = None,
) -> Callable:
    """
    Decorator that transparently caches a function's return value.

    The cache key is computed by *key_fn* (if supplied) or by joining the
    function's qualified name with the ``repr()`` of all positional and
    keyword arguments.

    Parameters
    ----------
    cache:
        The :class:`DiskCache` instance to use.
    key_fn:
        Optional callable ``(*args, **kwargs) -> str`` that produces the
        cache key.  If ``None``, a sensible default is generated from the
        function name and its arguments.

    Returns
    -------
    Callable
        The wrapped function with transparent caching.

    Examples
    --------
    ::

        store = get_cache("osm", ttl_seconds=86400)

        @cached(store)
        def fetch_road_network(city: str) -> nx.Graph:
            ...

        # Custom key function
        @cached(store, key_fn=lambda city, radius: f"osm:{city}:{radius}")
        def fetch_road_network_bounded(city: str, radius: float) -> nx.Graph:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                arg_repr = ", ".join(
                    [repr(a) for a in args]
                    + [f"{k}={v!r}" for k, v in sorted(kwargs.items())]
                )
                key = f"{func.__qualname__}({arg_repr})"

            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper

    return decorator
