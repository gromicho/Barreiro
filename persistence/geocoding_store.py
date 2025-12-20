# persistence/geocoding_store.py
"""Persistent geocoding cache per instance (Dropbox-backed).

This module intentionally separates:
- addresses list persistence (versioned snapshots) in persistence.dropbox_store
- geocoding cache persistence (append-only-ish JSONL) here

Backwards compatibility:
Callers still use:
    load_geocoding_cache(filename=...)
    save_geocoding_cache(cache, filename=..., payload=..., file_id=...)

Where 'filename' is interpreted as instance name (e.g. 'capelle').

Cache keys:
- By default, entries are keyed by normalize_address(address)
- Optionally bbox-sensitive via: normalize_address(address) + '|' + bbox_to_key(bbox)
"""

from __future__ import annotations

import json
import time

import dropbox

from persistence.dropbox_store import get_dropbox_client


class GeocodingStoreError(RuntimeError):
    """Raised when persistent geocoding cache operations fail."""


def normalize_address(address: str) -> str:
    """
    Normalize an address string to a stable cache key component.

    Args:
        address: Free text address.

    Returns:
        Normalized address string (lowercase, whitespace collapsed).
        Empty string if address is empty after stripping.
    """
    cleaned = str(address or '').strip().lower()
    if not cleaned:
        return ''
    return ' '.join(cleaned.split())


def _instance_name(filename: str) -> str:
    """
    Derive an instance name from a filename-ish string.

    Examples:
        'capelle' -> 'capelle'
        'capelle_addresses.json' -> 'capelle'
        '/instances/capelle/whatever.json' -> 'capelle'

    Args:
        filename: Instance identifier or path-like string.

    Returns:
        Instance name (never empty).
    """
    name = str(filename).strip().strip('/')
    if not name:
        return 'default'

    if '/' in name:
        parts = [p for p in name.split('/') if p]
        if 'instances' in parts:
            i = parts.index('instances')
            if i + 1 < len(parts):
                return parts[i + 1]
        return parts[-1].split('.')[0]

    base = name.split('.')[0]
    if base.endswith('_addresses'):
        base = base[: -len('_addresses')]
    return base or 'default'


def _cache_path(instance: str) -> str:
    """
    Dropbox path for the per-instance geocoding cache.

    Args:
        instance: Instance name.

    Returns:
        Dropbox path.
    """
    return f'/instances/{instance}/geocode_cache.jsonl'


def _download_text(dbx: dropbox.Dropbox, *, path: str) -> str | None:
    """
    Download a UTF-8 text file from Dropbox.

    Args:
        dbx: Dropbox client.
        path: Dropbox path.

    Returns:
        File content as text, or None if not found.

    Raises:
        dropbox.exceptions.ApiError: For non-not-found Dropbox API errors.
    """
    try:
        _md, res = dbx.files_download(path)
        return res.content.decode('utf-8', errors='replace')
    except dropbox.exceptions.ApiError as exc:
        msg = str(exc)
        if 'not_found' in msg.lower():
            return None
        raise


def _upload_text(dbx: dropbox.Dropbox, *, path: str, content: str) -> None:
    """
    Upload a UTF-8 text file to Dropbox (overwrite).

    Args:
        dbx: Dropbox client.
        path: Dropbox path.
        content: Text content.
    """
    dbx.files_upload(
        content.encode('utf-8'),
        path,
        mode=dropbox.files.WriteMode.overwrite,
        mute=True,
    )


def bbox_to_key(
    bbox: tuple[float, float, float, float] | None,
    *,
    decimals: int = 4,
) -> str:
    """
    Convert a bbox to a stable string key.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84, or None.
        decimals: Rounding precision for stability.

    Returns:
        Empty string if bbox is None, otherwise a rounded string key.
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


def _make_cache_key(
    address_norm: str,
    *,
    bbox: tuple[float, float, float, float] | None,
    include_bbox_in_key: bool,
) -> str:
    """
    Build the persistent-cache key.

    Args:
        address_norm: Normalized address.
        bbox: Optional bbox in WGS84.
        include_bbox_in_key: If True, key includes bbox_to_key(bbox).

    Returns:
        Cache key (never empty if address_norm is not empty).
    """
    if not include_bbox_in_key:
        return address_norm

    bkey = bbox_to_key(bbox)
    if not bkey:
        return address_norm
    return f'{address_norm}|{bkey}'


def cache_get(
    cache: dict[str, dict[str, object]],
    address_norm: str,
    *,
    bbox: tuple[float, float, float, float] | None,
    include_bbox_in_key: bool,
) -> tuple[float, float] | None:
    """
    Read from the persistent cache dict.

    Args:
        cache: Persistent cache mapping.
        address_norm: Normalized address (normalize_address(address)).
        bbox: Optional bbox in WGS84.
        include_bbox_in_key: If True, use bbox-sensitive key.

    Returns:
        (lat, lon) if present and parseable, otherwise None.
    """
    if not address_norm:
        return None

    key = _make_cache_key(address_norm, bbox=bbox, include_bbox_in_key=include_bbox_in_key)
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None

    try:
        lat = float(entry['lat'])
        lon = float(entry['lon'])
    except Exception:
        return None

    return lat, lon


def cache_set(
    cache: dict[str, dict[str, object]],
    address_norm: str,
    *,
    lat: float,
    lon: float,
    formatted_address: str,
    status: str,
    source: str,
    bbox: tuple[float, float, float, float] | None,
    include_bbox_in_key: bool,
) -> None:
    """
    Write an entry to the persistent cache dict.

    Notes:
        This module historically used the field name 'formatted'. To keep
        backward compatibility, we store both:
        - formatted (legacy)
        - formatted_address (newer callers)

    Args:
        cache: Persistent cache mapping (mutated in-place).
        address_norm: Normalized address.
        lat: Latitude.
        lon: Longitude.
        formatted_address: Provider formatted address.
        status: Provider status (e.g. 'OK').
        source: Provider name (e.g. 'google').
        bbox: Optional bbox in WGS84.
        include_bbox_in_key: If True, key includes bbox_to_key(bbox).
    """
    if not address_norm:
        return

    key = _make_cache_key(address_norm, bbox=bbox, include_bbox_in_key=include_bbox_in_key)
    entry: dict[str, object] = {
        'lat': float(lat),
        'lon': float(lon),
        'formatted': str(formatted_address or ''),
        'formatted_address': str(formatted_address or ''),
        'status': str(status or ''),
        'source': str(source or ''),
    }

    if include_bbox_in_key:
        entry['bbox_key'] = bbox_to_key(bbox)

    cache[key] = entry


def load_geocoding_cache(
    *,
    filename: str = 'capelle',
) -> tuple[dict[str, dict[str, object]], dict[str, object], str]:
    """
    Load per-instance geocoding cache from JSONL.

    Returns:
        (cache_dict, payload_like, file_id_like)

    cache_dict maps cache_key -> entry dict
        cache_key is either:
            address_norm
        or:
            address_norm|bbox_key

    payload_like is kept for backward compatibility and is always {}.
    file_id_like is the Dropbox path to the JSONL file.
    """
    instance = _instance_name(filename)
    path = _cache_path(instance)

    try:
        dbx = get_dropbox_client()
        content = _download_text(dbx, path=path)

        cache: dict[str, dict[str, object]] = {}
        if content:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                key = rec.get('address_norm')
                if not isinstance(key, str) or not key:
                    continue

                lat = rec.get('lat')
                lon = rec.get('lon')
                formatted = rec.get('formatted', rec.get('formatted_address', ''))
                status = rec.get('status', '')
                source = rec.get('source', '')
                bbox_key = rec.get('bbox_key', '')

                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    entry: dict[str, object] = {
                        'lat': float(lat),
                        'lon': float(lon),
                        'formatted': str(formatted) if isinstance(formatted, str) else '',
                        'formatted_address': str(formatted) if isinstance(formatted, str) else '',
                        'status': str(status) if isinstance(status, str) else '',
                        'source': str(source) if isinstance(source, str) else '',
                    }
                    if isinstance(bbox_key, str) and bbox_key:
                        entry['bbox_key'] = bbox_key
                    cache[key] = entry

        return cache, {}, path

    except Exception as exc:
        raise GeocodingStoreError(str(exc)) from exc


def save_geocoding_cache(
    cache: dict[str, dict[str, object]],
    *,
    filename: str = 'capelle',
    payload: dict[str, object] | None = None,
    file_id: str | None = None,
) -> str:
    """
    Persist the per-instance geocoding cache.

    Implementation detail:
        We rewrite the JSONL file from the in-memory cache. This is simple and
        robust for modest cache sizes. If this cache becomes very large, switch
        to upload sessions and true append semantics.

    Backwards compatibility:
        payload and file_id are accepted but not required. The returned value is
        the Dropbox path (file_id-like).

    Args:
        cache: Cache mapping to persist.
        filename: Instance identifier.
        payload: Unused (compat only).
        file_id: Unused (compat only).

    Returns:
        Dropbox path to the JSONL file.
    """
    _ = payload
    _ = file_id

    instance = _instance_name(filename)
    path = _cache_path(instance)

    try:
        dbx = get_dropbox_client()
        ts = int(time.time())

        lines: list[str] = []
        for cache_key in sorted(cache.keys()):
            rec_in = cache.get(cache_key, {})
            if not isinstance(rec_in, dict):
                continue

            lat = rec_in.get('lat')
            lon = rec_in.get('lon')
            formatted = rec_in.get('formatted', rec_in.get('formatted_address', ''))
            status = rec_in.get('status', '')
            source = rec_in.get('source', '')
            bbox_key = rec_in.get('bbox_key', '')

            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue

            rec_out: dict[str, object] = {
                'address_norm': str(cache_key),
                'lat': float(lat),
                'lon': float(lon),
                'formatted': str(formatted) if isinstance(formatted, str) else '',
                'status': str(status) if isinstance(status, str) else '',
                'source': str(source) if isinstance(source, str) else '',
                'ts': ts,
            }

            if isinstance(bbox_key, str) and bbox_key:
                rec_out['bbox_key'] = bbox_key

            lines.append(json.dumps(rec_out, ensure_ascii=True))

        _upload_text(dbx, path=path, content='\n'.join(lines) + ('\n' if lines else ''))
        return path

    except Exception as exc:
        raise GeocodingStoreError(str(exc)) from exc


def clear_geocoding_cache(*, filename: str = 'capelle') -> str:
    """
    Clear per-instance geocoding cache by overwriting JSONL with empty content.

    Args:
        filename: Instance identifier.

    Returns:
        Dropbox path to the JSONL file.
    """
    instance = _instance_name(filename)
    path = _cache_path(instance)

    try:
        dbx = get_dropbox_client()
        _upload_text(dbx, path=path, content='')
        return path
    except Exception as exc:
        raise GeocodingStoreError(str(exc)) from exc
