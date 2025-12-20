"""
Dropbox-backed persistence for Streamlit apps.

Stores a small JSON file (addresses + optional geocoding cache) in a Dropbox
App Folder.

Notes
-----
- Uses Dropbox OAuth refresh token flow (recommended for Streamlit Cloud).
- Works with private Dropbox accounts.
- Designed to work the same locally and on Streamlit Cloud.
- Reads configuration from Streamlit secrets first, then environment variables.

Required secrets/env
--------------------
Preferred (Streamlit secrets):
  [dropbox]
  app_key = "..."
  app_secret = "..."
  refresh_token = "..."

Environment variables fallback:
  DROPBOX_APP_KEY
  DROPBOX_APP_SECRET
  DROPBOX_REFRESH_TOKEN
"""

from __future__ import annotations

import json
import os
import time
from datetime import timezone

import dropbox


class DropboxStoreError(RuntimeError):
    """Raised when DropboxStore operations fail."""


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _get_secret(name: str, default: str = '') -> str:
    """
    Retrieve a secret from Streamlit secrets or environment variables.

    Supports:
      - st.secrets['dropbox'][key]   (preferred)
      - st.secrets[name]            (legacy/top-level)
      - os.environ[name]            (fallback)

    Args:
        name: Secret name, e.g. 'DROPBOX_APP_KEY'.
        default: Default value if not found.

    Returns:
        The secret value as a string.
    """
    try:
        import streamlit as st
    except Exception:
        st = None

    if st is not None:
        try:
            cfg = st.secrets.get('dropbox', None)
            if cfg is not None and hasattr(cfg, 'get'):
                key = name.lower().removeprefix('dropbox_')
                val = cfg.get(key, default)
                return val if isinstance(val, str) else str(val)
        except Exception:
            pass

        try:
            val = st.secrets.get(name, default)
            return val if isinstance(val, str) else str(val)
        except Exception:
            pass

    val_env = os.environ.get(name, default)
    return str(val_env) if val_env is not None else str(default)


def _now_iso_utc() -> str:
    """Return a compact ISO-like UTC timestamp string."""
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _build_dropbox_client_uncached() -> dropbox.Dropbox:
    """
    Create a Dropbox client using OAuth refresh token flow.

    Expects secrets/env:
      - DROPBOX_APP_KEY
      - DROPBOX_APP_SECRET
      - DROPBOX_REFRESH_TOKEN
    """
    app_key = _get_secret('DROPBOX_APP_KEY', '').strip()
    app_secret = _get_secret('DROPBOX_APP_SECRET', '').strip()
    refresh_token = _get_secret('DROPBOX_REFRESH_TOKEN', '').strip()

    missing: list[str] = []
    if not app_key:
        missing.append('DROPBOX_APP_KEY')
    if not app_secret:
        missing.append('DROPBOX_APP_SECRET')
    if not refresh_token:
        missing.append('DROPBOX_REFRESH_TOKEN')

    if missing:
        raise DropboxStoreError(
            'Missing Dropbox credentials in Streamlit secrets or environment: '
            + ', '.join(missing)
            + '.',
        )

    return dropbox.Dropbox(
        oauth2_refresh_token=refresh_token,
        app_key=app_key,
        app_secret=app_secret,
    )


def _build_dropbox_client() -> dropbox.Dropbox:
    """Create a Dropbox client, cached when Streamlit is available."""
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _cached() -> dropbox.Dropbox:
            return _build_dropbox_client_uncached()

        return _cached()
    except Exception:
        return _build_dropbox_client_uncached()


# ---------------------------------------------------------------------
# Dropbox file primitives
# ---------------------------------------------------------------------


def _dbx_path(path: str) -> str:
    """Normalize Dropbox paths (App Folder-relative)."""
    if not path.startswith('/'):
        return '/' + path
    return path


def _dropbox_download_text(dbx: dropbox.Dropbox, *, path: str) -> str:
    """Download a text file from Dropbox."""
    try:
        _md, res = dbx.files_download(_dbx_path(path))
        return res.content.decode('utf-8', errors='replace')
    except dropbox.exceptions.ApiError as exc:
        raise DropboxStoreError(f'Failed to download "{path}": {exc}') from exc


def _dropbox_upload_text(
    dbx: dropbox.Dropbox,
    *,
    path: str,
    content: str,
) -> None:
    """Upload or overwrite a text file in Dropbox."""
    try:
        dbx.files_upload(
            content.encode('utf-8'),
            _dbx_path(path),
            mode=dropbox.files.WriteMode.overwrite,
        )
    except dropbox.exceptions.ApiError as exc:
        raise DropboxStoreError(f'Failed to upload "{path}": {exc}') from exc


def _is_dropbox_path_not_found(exc: dropbox.exceptions.ApiError) -> bool:
    """Return True if the ApiError corresponds to a missing path."""
    try:
        err = exc.error
        if hasattr(err, 'is_path') and err.is_path():
            path_err = err.get_path()
            return hasattr(path_err, 'is_not_found') and path_err.is_not_found()
    except Exception:
        return False
    return False


def _dropbox_exists(dbx: dropbox.Dropbox, *, path: str) -> bool:
    """Check whether a file exists in Dropbox."""
    try:
        dbx.files_get_metadata(_dbx_path(path))
        return True
    except dropbox.exceptions.ApiError as exc:
        if _is_dropbox_path_not_found(exc):
            return False
        raise DropboxStoreError(f'Failed to stat "{path}": {exc}') from exc
    return False


def _dropbox_list_folder_files(
    dbx: dropbox.Dropbox,
    *,
    folder: str,
) -> list[dropbox.files.FileMetadata]:
    """List files in a Dropbox folder (non-recursive)."""
    folder_norm = _dbx_path(folder)
    try:
        res = dbx.files_list_folder(folder_norm, recursive=False)
        entries: list[dropbox.files.FileMetadata] = []
        while True:
            for e in res.entries:
                if isinstance(e, dropbox.files.FileMetadata):
                    entries.append(e)
            if not res.has_more:
                break
            res = dbx.files_list_folder_continue(res.cursor)
        return entries
    except dropbox.exceptions.ApiError as exc:
        if _is_dropbox_path_not_found(exc):
            return []
        raise DropboxStoreError(f'Failed to list folder "{folder}": {exc}') from exc


def _fmt_dropbox_dt_utc(dt_obj: object) -> str:
    """Format Dropbox datetime-like object as 'YYYY-MM-DD HH:MM UTC'."""
    try:
        if hasattr(dt_obj, 'astimezone'):
            dt_utc = dt_obj.astimezone(timezone.utc)
            return dt_utc.strftime('%Y-%m-%d %H:%M UTC')
    except Exception:
        pass
    return ''


# ---------------------------------------------------------------------
# store logic
# ---------------------------------------------------------------------


def get_dropbox_client() -> dropbox.Dropbox:
    """Get a Dropbox client."""
    return _build_dropbox_client()


def default_payload() -> dict[str, object]:
    """Create the default JSON payload stored in Dropbox."""
    return {
        'version': 1,
        'updated_at': _now_iso_utc(),
        'addresses_text': '',
        'address_count': 0,
        'geocoding_cache': {},
    }


def load_or_init_store(
    *,
    path: str = 'capelle_addresses.json',
) -> tuple[dict[str, object], str]:
    """
    Load the JSON store from Dropbox, or return a default payload if missing.

    Args:
        path: Path inside the Dropbox App Folder.

    Returns:
        (payload, path)
    """
    dbx = _build_dropbox_client()
    path_norm = _dbx_path(path)

    if not _dropbox_exists(dbx, path=path_norm):
        return default_payload(), path_norm

    try:
        text = _dropbox_download_text(dbx, path=path_norm)
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError('Payload is not a JSON object.')
    except Exception as exc:
        raise DropboxStoreError(
            f'Failed to download/parse store file "{path_norm}": {exc}',
        ) from exc

    base = default_payload()
    for key, value in base.items():
        payload.setdefault(key, value)

    return payload, path_norm


def save_store(
    payload: dict[str, object],
    *,
    path: str = 'capelle_addresses.json',
) -> str:
    """
    Save the JSON payload to Dropbox (overwrite).

    Args:
        payload: JSON-serializable dict to store.
        path: Path inside the Dropbox App Folder.

    Returns:
        The Dropbox path.
    """
    dbx = _build_dropbox_client()
    return _save_store_at_path(dbx, payload=payload, path=path)


# ---------------------------------------------------------------------
# convenience API (same shape as your Drive version)
# ---------------------------------------------------------------------


def _instance_name(filename: str) -> str:
    """Derive instance name from a legacy filename argument."""
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


def _instance_addresses_path(instance: str) -> str:
    """Dropbox path for the latest addresses snapshot JSON."""
    return f'instances/{instance}/addresses.json'


def _instance_addresses_version_path(instance: str, version: int) -> str:
    """Dropbox path for a versioned addresses snapshot JSON."""
    return f'instances/{instance}/addresses.{version}.json'


def _count_addresses(addresses_text: str) -> int:
    """Count non-empty address lines in the addresses_text payload."""
    lines = str(addresses_text or '').splitlines()
    return int(sum(1 for line in lines if line.strip()))


def _dropbox_mkdirs(dbx: dropbox.Dropbox, *, folder: str) -> None:
    """Ensure a folder exists in Dropbox (App Folder)."""
    folder_norm = _dbx_path(folder)
    try:
        dbx.files_create_folder_v2(folder_norm)
    except dropbox.exceptions.ApiError as exc:
        try:
            err = exc.error
            if hasattr(err, 'is_path') and err.is_path():
                path_err = err.get_path()
                if hasattr(path_err, 'is_conflict') and path_err.is_conflict():
                    return
        except Exception:
            pass
        raise DropboxStoreError(f'Failed to create folder "{folder}": {exc}') from exc


def _save_store_at_path(dbx: dropbox.Dropbox, *, payload: dict[str, object], path: str) -> str:
    """Save JSON payload to a specific Dropbox path (overwrite)."""
    path_norm = _dbx_path(path)

    payload_out = dict(payload)
    payload_out['updated_at'] = _now_iso_utc()

    try:
        content = json.dumps(payload_out, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise DropboxStoreError(f'Payload is not JSON-serializable: {exc}') from exc

    _dropbox_upload_text(dbx, path=path_norm, content=content)
    return path_norm


def load_addresses_text(
    *,
    filename: str = 'capelle',
) -> tuple[str, dict[str, object], str | None]:
    """
    Load the latest address list for an instance.

    Returns:
        (addresses_text, payload, file_id_like)
    """
    instance = _instance_name(filename)
    path = _instance_addresses_path(instance)

    payload, path_norm = load_or_init_store(path=path)
    text = str(payload.get('addresses_text', '') or '')
    return text, payload, path_norm


def list_address_versions(*, filename: str = 'capelle') -> list[dict[str, object]]:
    """List available versioned address snapshots for an instance.

    Returns:
        A list of dicts with keys:
          - version: int
          - timestamp: str (e.g. '2025-12-17 17:11 UTC')
          - path: str (Dropbox path)
          - address_count: int | None
    """
    instance = _instance_name(filename)
    folder = f'instances/{instance}'
    dbx = _build_dropbox_client()
    files = _dropbox_list_folder_files(dbx, folder=folder)

    out: list[dict[str, object]] = []
    prefix = 'addresses.'
    suffix = '.json'

    for meta in files:
        name = getattr(meta, 'name', '')
        if not (isinstance(name, str) and name.startswith(prefix) and name.endswith(suffix)):
            continue

        mid = name[len(prefix):-len(suffix)]
        if not mid.isdigit():
            continue

        version = int(mid)
        ts = _fmt_dropbox_dt_utc(getattr(meta, 'server_modified', None))
        path_display = f'{folder}/{name}'

        address_count: int | None = None
        try:
            text = _dropbox_download_text(dbx, path=path_display)
            payload = json.loads(text)
            if isinstance(payload, dict):
                raw = payload.get('address_count', None)
                if isinstance(raw, int):
                    address_count = raw
        except Exception:
            address_count = None

        out.append(
            {
                'version': version,
                'timestamp': ts,
                'path': path_display,
                'address_count': address_count,
            }
        )

    out.sort(key=lambda d: int(d.get('version', 0)), reverse=True)
    return out


def save_addresses_text(
    addresses_text: str,
    *,
    filename: str = 'capelle',
    payload: dict[str, object] | None = None,
    file_id: str | None = None,
) -> str:
    """Save addresses text as a new versioned snapshot and update latest pointer.

    Args:
        addresses_text: Raw address text, typically one address per line.
        filename: Logical store name used to derive the instance folder.
        payload: Optional base payload to extend or override.
        file_id: Optional identifier used by older backends (unused for Dropbox).

    Returns:
        Dropbox path of the newly created versioned snapshot.
    """
    _ = file_id

    normalized_text = str(addresses_text)
    address_count = _count_addresses(normalized_text)

    versions = list_address_versions(filename=filename)
    if versions:
        next_version = max(int(v.get('version', 0)) for v in versions) + 1
    else:
        next_version = 1

    payload_out: dict[str, object] = default_payload()
    if payload is not None:
        if not isinstance(payload, dict):
            raise TypeError('payload must be a dict[str, object] or None')
        payload_out.update(payload)

    payload_out.update(
        {
            'version': int(next_version),
            'addresses_text': normalized_text,
            'address_count': int(address_count),
        }
    )

    instance = _instance_name(filename)
    folder = f'instances/{instance}'
    versioned_path = _instance_addresses_version_path(instance, int(next_version))
    latest_path = _instance_addresses_path(instance)

    dbx = _build_dropbox_client()
    _dropbox_mkdirs(dbx, folder=folder)

    _save_store_at_path(dbx, payload=payload_out, path=versioned_path)
    _save_store_at_path(dbx, payload=payload_out, path=latest_path)

    return _dbx_path(versioned_path)


def load_addresses_text_version(
    *,
    filename: str = 'capelle',
    version: int,
) -> tuple[str, dict[str, object], str | None]:
    """Load a specific versioned address snapshot for an instance."""
    instance = _instance_name(filename)
    path = _instance_addresses_version_path(instance, int(version))
    payload, path_norm = load_or_init_store(path=path)
    text = str(payload.get('addresses_text', '') or '')
    return text, payload, path_norm


def load_addresses_only(*, filename: str = 'capelle') -> str:
    """Convenience: load only addresses_text."""
    text, _payload, _path = load_addresses_text(filename=filename)
    return text
