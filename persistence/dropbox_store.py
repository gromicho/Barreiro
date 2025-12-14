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
        # Preferred: [dropbox] section in secrets.toml
        try:
            cfg = st.secrets.get('dropbox', None)
            if cfg is not None and hasattr(cfg, 'get'):
                key = name.lower().removeprefix('dropbox_')
                val = cfg.get(key, default)
                if isinstance(val, str):
                    return val
                return str(val)
        except Exception:
            pass

        # Legacy: top-level secret
        try:
            val = st.secrets.get(name, default)
            if isinstance(val, str):
                return val
            return str(val)
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


# ---------------------------------------------------------------------
# store logic
# ---------------------------------------------------------------------


def default_payload() -> dict[str, object]:
    """Create the default JSON payload stored in Dropbox."""
    return {
        'version': 1,
        'updated_at': _now_iso_utc(),
        'addresses_text': '',
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
    path_norm = _dbx_path(path)

    payload_out = dict(payload)
    payload_out['updated_at'] = _now_iso_utc()

    try:
        content = json.dumps(payload_out, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise DropboxStoreError(f'Payload is not JSON-serializable: {exc}') from exc

    _dropbox_upload_text(dbx, path=path_norm, content=content)
    return path_norm


# ---------------------------------------------------------------------
# convenience API (same shape as your Drive version)
# ---------------------------------------------------------------------


def load_addresses_text(
    *,
    filename: str = 'capelle_addresses.json',
) -> tuple[str, dict[str, object], str | None]:
    """
    Backwards-compatible: return (addresses_text, payload, file_id_like).

    For Dropbox, the third return value is the Dropbox path (string). We keep the
    name 'file_id' in callers by returning it here.
    """
    payload, path = load_or_init_store(path=filename)
    return str(payload.get('addresses_text', '')), payload, path


def save_addresses_text(
    addresses_text: str,
    *,
    filename: str = 'capelle_addresses.json',
    payload: dict[str, object] | None = None,
    file_id: str | None = None,
) -> str:
    """
    Backwards-compatible wrapper for Drive-era callers.

    Args:
        addresses_text: Text area content.
        filename: Store filename (Dropbox path if file_id is None).
        payload: Existing payload to update; if None, load/init first.
        file_id: Drive-era identifier; treated as Dropbox path when provided.

    Returns:
        Dropbox path (string).
    """
    path = str(file_id).strip() if file_id else filename

    payload_in = payload
    if payload_in is None:
        payload_in, _path_loaded = load_or_init_store(path=path)

    payload_out = dict(payload_in)
    payload_out['addresses_text'] = str(addresses_text)

    return save_store(payload_out, path=path)


def load_addresses_only(*, filename: str = 'capelle_addresses.json') -> str:
    """Convenience: load only addresses_text."""
    text, _payload, _path = load_addresses_text(filename=filename)
    return text
