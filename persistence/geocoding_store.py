# geocoding_store.py
"""
Geocoding cache persistence on top of dropbox_store.py.

This module:
- normalizes addresses (as cache keys),
- stores per-address geocoding results in Dropbox JSON payload['geocoding_cache'],
- provides small helpers to use the cache during geocoding runs.

Cache design
------------

Key:
- normalized address string (lowercased, whitespace-collapsed)

Value (dict):
- lat: float
- lon: float
- formatted_address: str (optional)
- status: str (e.g., 'OK', 'ZERO_RESULTS', 'ERROR')
- source: str (e.g., 'google')
- updated_at: str (timestamp)
- bbox_key: str (optional, to separate results by geocoding bounds)

Notes
-----
- If you want bbox-sensitive caching: enable bbox_key generation by passing bbox
  and setting include_bbox_in_key=True.
- The module does not call Google itself. It only persists/cache-wraps your
  existing geocoding function(s).
"""

from __future__ import annotations

import math
import time

from persistence.dropbox_store import (
    DropboxStoreError,
    load_or_init_store,
    save_store,
)


class GeocodingStoreError(RuntimeError):
    """Raised when geocoding-store operations fail."""


def _now_iso_utc() -> str:
    """Return a compact ISO-like UTC timestamp string."""
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def normalize_address(address: str) -> str:
    """
    Normalize an address to a stable cache key.

    Rules:
    - strip leading/trailing whitespace
    - collapse internal whitespace to single spaces
    - lowercase
    """
    parts = str(address).strip().split()
    if not parts:
        return ''
    return ' '.join(parts).lower()


def normalize_addresses_text(text: str) -> str:
    """
    Normalize an addresses text blob: trim lines, drop empties, join with '\\n', end with '\\n'.
    """
    lines = [ln.strip() for ln in str(text).splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ''
    return '\n'.join(lines) + '\n'


def bbox_to_key(
    bbox: tuple[float, float, float, float] | None,
    *,
    decimals: int = 4,
) -> str:
    """
    Convert a bbox to a stable string key.

    bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
    """
    if bbox is None:
        return ''

    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        f'{round(float(min_lon), decimals)},'
        f'{round(float(min_lat), decimals)},'
        f'{round(float(max_lon), decimals)},'
        f'{round(float(max_lat), decimals)}'
    )


def _ensure_payload_schema(payload: dict[str, object]) -> dict[str, object]:
    """Ensure required keys exist in payload."""
    out = dict(payload)

    cache = out.get('geocoding_cache')
    if not isinstance(cache, dict):
        out['geocoding_cache'] = {}

    if 'addresses_text' not in out:
        out['addresses_text'] = ''
    if 'version' not in out:
        out['version'] = 1
    if 'updated_at' not in out:
        out['updated_at'] = _now_iso_utc()

    return out


def load_geocoding_cache(
    *,
    filename: str = 'capelle_addresses.json',
) -> tuple[dict[str, dict[str, object]], dict[str, object], str]:
    """
    Load the Dropbox store and return (geocoding_cache, payload, path).

    Returns:
        (cache, payload, path)

        - cache: dict[normalized_address, entry_dict]
        - payload: full store payload
        - path: Dropbox path (store identifier)
    """
    try:
        payload, path = load_or_init_store(path=filename)
        payload = _ensure_payload_schema(payload)

        cache_obj = payload.get('geocoding_cache', {})
        if not isinstance(cache_obj, dict):
            cache_obj = {}
            payload['geocoding_cache'] = cache_obj

        cache_typed: dict[str, dict[str, object]] = {}
        for k, v in cache_obj.items():
            if isinstance(k, str) and isinstance(v, dict):
                cache_typed[k] = v

        return cache_typed, payload, path

    except DropboxStoreError as exc:
        raise GeocodingStoreError(str(exc)) from exc
    except Exception as exc:
        raise GeocodingStoreError(f'Failed to load geocoding cache: {exc}') from exc


def save_geocoding_cache(
    cache: dict[str, dict[str, object]],
    *,
    payload: dict[str, object] | None = None,
    path: str | None = None,
    filename: str = 'capelle_addresses.json',
) -> str:
    """
    Save a geocoding cache dict back into the Dropbox store.

    Args:
        cache: Cache dict.
        payload: Existing payload; if None, it will be loaded.
        path: Existing Dropbox path; optional.
        filename: Store filename.

    Returns:
        Dropbox path.
    """
    try:
        if payload is None:
            _cache_old, payload_loaded, path_loaded = load_geocoding_cache(filename=filename)
            payload = payload_loaded
            if path is None:
                path = path_loaded

        payload_out = _ensure_payload_schema(payload)
        payload_out['geocoding_cache'] = dict(cache)
        payload_out['updated_at'] = _now_iso_utc()

        store_path = path or filename
        return save_store(payload_out, path=store_path)

    except DropboxStoreError as exc:
        raise GeocodingStoreError(str(exc)) from exc
    except Exception as exc:
        raise GeocodingStoreError(f'Failed to save geocoding cache: {exc}') from exc


def cache_get(
    cache: dict[str, dict[str, object]],
    address: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    include_bbox_in_key: bool = False,
) -> tuple[float, float] | None:
    """Get cached (lat, lon) for an address, if present."""
    key = normalize_address(address)
    if not key:
        return None

    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None

    if include_bbox_in_key:
        want = bbox_to_key(bbox)
        have = str(entry.get('bbox_key', ''))
        if want != have:
            return None

    try:
        lat = float(entry.get('lat'))
        lon = float(entry.get('lon'))
    except Exception:
        return None

    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None

    return lat, lon


def cache_set(
    cache: dict[str, dict[str, object]],
    address: str,
    *,
    lat: float,
    lon: float,
    formatted_address: str = '',
    status: str = 'OK',
    source: str = 'google',
    bbox: tuple[float, float, float, float] | None = None,
    include_bbox_in_key: bool = False,
) -> None:
    """Set cached geocoding result for an address (mutates cache in place)."""
    key = normalize_address(address)
    if not key:
        return

    entry: dict[str, object] = {
        'lat': float(lat),
        'lon': float(lon),
        'formatted_address': str(formatted_address),
        'status': str(status),
        'source': str(source),
        'updated_at': _now_iso_utc(),
    }
    if include_bbox_in_key:
        entry['bbox_key'] = bbox_to_key(bbox)

    cache[key] = entry
