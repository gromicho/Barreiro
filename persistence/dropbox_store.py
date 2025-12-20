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
        # Dropbox returns aware datetime in most SDKs.
        dt = dt_obj
        if hasattr(dt, 'astimezone'):
            dt_utc = dt.astimezone(timezone.utc)
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



def _instance_name(filename: str) -> str:
    """Derive instance name from a legacy filename argument.

    Historically, callers passed values like 'capelle_addresses.json'.
    In the reengineered layout, callers should pass just 'capelle' or
    'barreiro'. We accept both.

    Examples:
        'capelle' -> 'capelle'
        'capelle_addresses.json' -> 'capelle'
        'instances/capelle/addresses.json' -> 'capelle'
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

    base = name
    base = base.split('.')[0]
    if base.endswith('_addresses'):
        base = base[: -len('_addresses')]
    return base or 'default'


def _instance_addresses_path(instance: str) -> str:
    """Dropbox path for the latest addresses snapshot JSON."""
    return f'instances/{instance}/addresses.json'


def _instance_addresses_version_path(instance: str, version: int) -> str:
    """Dropbox path for a versioned addresses snapshot JSON."""
    return f'instances/{instance}/addresses.{version}.json'


def load_addresses_text(
    *,
    filename: str = 'capelle',
) -> tuple[str, dict[str, object], str | None]:
    """
    Load the latest address list for an instance.

    Backwards-compatible return shape:
        (addresses_text, payload, file_id_like)

    Notes:
        - 'filename' is interpreted as instance name (e.g. 'capelle').
        - payload includes 'version' and 'addresses_text' keys.
        - file_id_like is the Dropbox path to the latest snapshot.
    """
    instance = _instance_name(filename)
    path = _instance_addresses_path(instance)

    try:
        payload, path_norm = load_or_init_store(path=path)
    except DropboxStoreError as exc:
        raise DropboxStoreError(str(exc)) from exc

    text = str(payload.get('addresses_text', '') or '')
    return text, payload, path_norm


def save_addresses_text(
    addresses_text: str,
    *,
    filename: str = 'capelle',
    payload: dict[str, object] | None = None,
    file_id: str | None = None,
) -> str:
    """
    Save addresses_text as a new versioned snapshot for an instance.

    Versioning:
        - Reads current latest 'instances/<instance>/addresses.json'
        - Next version is (current_version + 1), defaulting to 1
        - Writes:
            - instances/<instance>/addresses.<version>.json
            - instances/<instance>/addresses.json

    Returns:
        Dropbox path to the latest snapshot JSON.
    """
    instance = _instance_name(filename)
    latest_path = _instance_addresses_path(instance)

    current_payload, _path_norm = load_or_init_store(path=latest_path)
    current_version = current_payload.get('version')
    if isinstance(current_version, int) and current_version >= 0:
        next_version = current_version + 1
    else:
        next_version = 1

    payload_out: dict[str, object] = {
        'version': next_version,
        'addresses_text': str(addresses_text),
    }

    version_path = _instance_addresses_version_path(instance, next_version)
    save_store(payload_out, path=version_path)
    return save_store(payload_out, path=latest_path)


def load_addresses_only(*, filename: str = 'capelle') -> str:
    """Convenience: load only addresses_text."""
    text, _payload, _path = load_addresses_text(filename=filename)
    return text


def list_address_versions(*, filename: str = 'capelle') -> list[dict[str, object]]:
    """List available versioned address snapshots for an instance.

    Returns:
        A list of dicts with keys:
          - version: int
          - timestamp: str (e.g. '2025-12-17 17:11 UTC')
          - path: str (Dropbox path)

    Notes:
        Uses Dropbox file metadata timestamps (server_modified) to avoid
        downloading all snapshots.
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
        out.append({'version': version, 'timestamp': ts, 'path': path_display})

    out.sort(key=lambda d: int(d.get('version', 0)), reverse=True)
    return out


def load_addresses_text_version(
    *,
    filename: str = 'capelle',
    version: int,
) -> tuple[str, dict[str, object], str | None]:
    """Load a specific versioned address snapshot for an instance.

    Args:
        filename: Instance name (or legacy filename).
        version: Snapshot version number.

    Returns:
        (addresses_text, payload, file_id_like)
    """
    instance = _instance_name(filename)
    path = _instance_addresses_version_path(instance, int(version))
    payload, path_norm = load_or_init_store(path=path)
    text = str(payload.get('addresses_text', '') or '')
    return text, payload, path_norm
