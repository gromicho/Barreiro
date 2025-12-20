# geocoder.py
"""
Google Geocoding helpers for Streamlit apps.

Responsibilities:
- Forward geocode addresses using Google Geocoding API.
- Provide a robust, testable cache layer:
  - In-memory (Streamlit cache_data) for speed within a running app instance.
  - Persistent (Dropbox JSON store) via persistence.geocoding_store.

Design goals:
- Works the same locally and on Streamlit Cloud.
- Avoids side effects at import time (no Drive calls until functions are used).

Secrets supported:
- GOOGLE_MAPS_API_KEY (required)
"""

import os
import time

import requests
import streamlit as st

from persistence.geocoding_store import (
    GeocodingStoreError,
    bbox_to_key,
    cache_get,
    cache_set,
    load_geocoding_cache,
    normalize_address,
    save_geocoding_cache,
)


class GeocodingError(RuntimeError):
    """Raised when geocoding fails."""


def get_google_maps_api_key() -> str:
    """
    Retrieve the Google Maps API key from Streamlit secrets or the environment.

    Returns:
        API key.

    Raises:
        GeocodingError: If not configured.
    """
    try:
        api_key = str(st.secrets.get('GOOGLE_MAPS_API_KEY', '')).strip()
    except Exception:
        api_key = ''

    if not api_key:
        api_key = str(os.environ.get('GOOGLE_MAPS_API_KEY', '')).strip()

    if not api_key:
        raise GeocodingError(
            'Google Maps API key not found. Set GOOGLE_MAPS_API_KEY in '
            '.streamlit/secrets.toml (or Streamlit Cloud secrets).'
        )

    return api_key


def _google_geocode_request(
    *,
    address: str,
    bbox: tuple[float, float, float, float] | None,
    timeout_s: int = 10,
) -> tuple[float, float, str]:
    """
    Call Google Geocoding API once.

    Args:
        address: Free text address.
        bbox: Optional (min_lon, min_lat, max_lon, max_lat) in WGS84.
        timeout_s: Requests timeout.

    Returns:
        (lat, lon, formatted_address)

    Raises:
        GeocodingError: On API errors or no results.
    """
    api_key = get_google_maps_api_key()
    cleaned = str(address).strip()
    if not cleaned:
        raise GeocodingError('Empty address.')

    url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params: dict[str, str] = {'address': cleaned, 'key': api_key}

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        sw = f'{float(min_lat)},{float(min_lon)}'
        ne = f'{float(max_lat)},{float(max_lon)}'
        params['bounds'] = f'{sw}|{ne}'

    try:
        resp = requests.get(url, params=params, timeout=int(timeout_s))
        resp.raise_for_status()
    except Exception as exc:
        raise GeocodingError(f'Google geocoding request failed: {exc}') from exc

    try:
        data = resp.json()
    except Exception as exc:
        raise GeocodingError(f'Google geocoding returned non-JSON: {exc}') from exc

    status = str(data.get('status', '') or '')
    if status != 'OK':
        msg = str(data.get('error_message', '') or '')
        if status == 'ZERO_RESULTS':
            raise GeocodingError('Google geocoding returned ZERO_RESULTS for this address.')
        raise GeocodingError(f'Google geocoding failed, status={status}, message={msg}')

    results = data.get('results', []) or []
    if not results:
        raise GeocodingError('Google geocoding returned no results.')

    try:
        loc = results[0]['geometry']['location']
        lat = float(loc['lat'])
        lon = float(loc['lng'])
    except Exception as exc:
        raise GeocodingError(f'Unexpected Google geocoding response shape: {exc}') from exc

    formatted = str(results[0].get('formatted_address', '') or '')
    return lat, lon, formatted


@st.cache_data(show_spinner=False)
def _geocode_in_memory_cached(
    address_norm: str,
    bbox_key: str,
    bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, str]:
    """
    In-memory cached wrapper around the Google API call.

    Note:
        bbox_key is included as an explicit argument so Streamlit caching differentiates
        between different bounds even if bbox is numerically "equivalent" via rounding.

    Args:
        address_norm: Normalized address string.
        bbox_key: Stable string key for bbox (possibly empty).
        bbox: Optional bbox in WGS84.

    Returns:
        (lat, lon, formatted_address)
    """
    _ = bbox_key
    return _google_geocode_request(address=address_norm, bbox=bbox)


def geocode_address(
    *,
    address: str,
    bbox: tuple[float, float, float, float] | None = None,
    persist: bool = True,
    store_filename: str = 'capelle_addresses.json',
    include_bbox_in_key: bool = False,
    throttle_s: float = 0.0,
) -> tuple[float, float]:
    """
    Geocode one address with a 2-level cache (Drive persistent + in-memory).

    Order:
    1) If persist=True, try Drive cache (keyed by normalized address, optionally bbox_key).
    2) Try Streamlit in-memory cache (calls Google once per unique key per process).
    3) If persist=True, write result back to Drive cache.

    Args:
        address: Address string.
        bbox: Optional bbox in WGS84.
        persist: Enable Dropbox persistent caching.
        store_filename: The Drive JSON store filename.
        include_bbox_in_key: If True, make persistent caching bbox-sensitive.
        throttle_s: Optional sleep before calling Google (cache miss only).

    Returns:
        (lat, lon) WGS84

    Raises:
        GeocodingError: If geocoding fails.
    """
    addr_norm = normalize_address(address)
    if not addr_norm:
        raise GeocodingError('Empty address.')

    bbox_key = bbox_to_key(bbox)

    payload: dict[str, object] | None = None
    file_id: str | None = None
    cache: dict[str, dict[str, object]] = {}

    if persist:
        try:
            cache, payload, path = load_geocoding_cache(filename=store_filename)
            cached = cache_get(
                cache,
                addr_norm,
                bbox=bbox,
                include_bbox_in_key=include_bbox_in_key,
            )
            if cached is not None:
                return cached
        except GeocodingStoreError:
            payload = None
            file_id = None
            cache = {}
        except Exception:
            payload = None
            file_id = None
            cache = {}

    if throttle_s > 0:
        time.sleep(float(throttle_s))

    lat, lon, formatted = _geocode_in_memory_cached(addr_norm, bbox_key, bbox)

    if persist:
        try:
            cache_set(
                cache,
                addr_norm,
                lat=float(lat),
                lon=float(lon),
                formatted_address=str(formatted),
                status='OK',
                source='google',
                bbox=bbox,
                include_bbox_in_key=include_bbox_in_key,
            )
            save_geocoding_cache(
                cache,
                payload=payload,
                path=path,
                filename=store_filename,
            )
        except GeocodingStoreError:
            pass
        except Exception:
            pass

    return float(lat), float(lon)


def geocode_addresses(
    *,
    addresses: list[str],
    bbox: tuple[float, float, float, float] | None,
    persist: bool = True,
    store_filename: str = 'capelle_addresses.json',
    include_bbox_in_key: bool = False,
    throttle_s: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Geocode a list of addresses.

    Args:
        addresses: List of address strings.
        bbox: Optional bbox in WGS84.
        persist: Enable persistent caching.
        store_filename: Drive JSON store filename.
        include_bbox_in_key: If True, make persistent caching bbox-sensitive.
        throttle_s: Optional sleep before each Google call (cache miss only).

    Returns:
        List of (lat, lon) in input order.

    Raises:
        GeocodingError: If any address fails.
    """
    coords: list[tuple[float, float]] = []
    for addr in addresses:
        lat, lon = geocode_address(
            address=addr,
            bbox=bbox,
            persist=persist,
            store_filename=store_filename,
            include_bbox_in_key=include_bbox_in_key,
            throttle_s=throttle_s,
        )
        coords.append((lat, lon))
    return coords


def cache_diagnostics(
    *,
    addresses: list[str],
    bbox: tuple[float, float, float, float] | None,
    store_filename: str = 'capelle_addresses.json',
    include_bbox_in_key: bool = False,
) -> dict[str, int]:
    """
    Provide simple cache diagnostics (counts) for UI.

    Counts how many of the given addresses would be Drive-cache hits.

    Args:
        addresses: Addresses to check.
        bbox: Optional bbox.
        store_filename: Drive JSON store filename.
        include_bbox_in_key: If True, check bbox-sensitive hits.

    Returns:
        Dict with keys: drive_hits, drive_misses, total
    """
    total = len(addresses)
    if total == 0:
        return {'drive_hits': 0, 'drive_misses': 0, 'total': 0}

    try:
        cache, _payload, _file_id = load_geocoding_cache(filename=store_filename)
    except Exception:
        return {'drive_hits': 0, 'drive_misses': total, 'total': total}

    hits = 0
    for addr in addresses:
        addr_norm = normalize_address(addr)
        if not addr_norm:
            continue
        if cache_get(cache, addr_norm, bbox=bbox, include_bbox_in_key=include_bbox_in_key) is not None:
            hits += 1

    return {'drive_hits': hits, 'drive_misses': total - hits, 'total': total}
