"""
Explicit Drive handlers (no Streamlit widgets).

These functions may touch Drive, but only when called.
"""

from persistence.dropbox_store import (
    DropboxStoreError,
    load_addresses_text,
    save_addresses_text,
)
from persistence.geocoding_store import (
    GeocodingStoreError,
    load_geocoding_cache,
    save_geocoding_cache,
)

from ui.errors import UiStateError
from ui.state_accessors import (
    get_addresses_text,
    get_drive_file_id,
    get_drive_payload,
    get_store_filename,
    normalize_addresses_text,
    set_addresses_text,
    set_drive_file_id,
    set_drive_payload,
)
from ui.state_keys import init_state_if_missing


def ensure_addresses_loaded(
    *,
    default_text: str,
    filename: str = 'capelle_addresses.json',
) -> None:
    """
    Load addresses_text from Drive store into session_state once, if missing/empty.

    This is intended for "first page load" behavior.

    Args:
        default_text: Used if Drive has no stored addresses (or Drive unavailable).
        filename: Drive JSON store filename.
    """
    init_state_if_missing(filename=filename)

    current = get_addresses_text()
    if current.strip():
        return

    try:
        addresses_text, payload, file_id = load_addresses_text(filename=filename)
        normalized = normalize_addresses_text(addresses_text) or normalize_addresses_text(default_text)

        set_addresses_text(normalized)
        set_drive_payload(payload if isinstance(payload, dict) else {})
        set_drive_file_id(file_id)
    except DropboxStoreError:
        set_addresses_text(normalize_addresses_text(default_text))
        set_drive_payload({})
        set_drive_file_id(None)
    except Exception:
        set_addresses_text(normalize_addresses_text(default_text))
        set_drive_payload({})
        set_drive_file_id(None)


def save_addresses_to_drive() -> None:
    """
    Save addresses_text to Drive, update stored file_id in session_state.

    Raises:
        UiStateError: On failures or empty list.
    """
    filename = get_store_filename()

    raw = get_addresses_text()
    normalized = normalize_addresses_text(raw)
    if not normalized.strip():
        raise UiStateError('Niets om op te slaan (lijst is leeg).')

    payload = get_drive_payload()
    file_id = get_drive_file_id()

    try:
        new_file_id = save_addresses_text(
            normalized,
            filename=filename,
            payload=payload,
            file_id=file_id,
        )
        set_addresses_text(normalized)
        set_drive_file_id(new_file_id)
        set_drive_payload(payload)
    except DropboxStoreError as exc:
        raise UiStateError(f'Opslaan mislukt: {exc}') from exc
    except Exception as exc:
        raise UiStateError(f'Opslaan mislukt: {exc}') from exc


def reload_addresses_from_drive(*, default_text: str) -> None:
    """
    Reload addresses_text from Drive into session_state.

    Args:
        default_text: Fallback if Drive is empty.

    Raises:
        UiStateError: On failures.
    """
    filename = get_store_filename()

    try:
        addresses_text, payload, file_id = load_addresses_text(filename=filename)
        normalized = normalize_addresses_text(addresses_text) or normalize_addresses_text(default_text)

        set_addresses_text(normalized)
        set_drive_payload(payload if isinstance(payload, dict) else {})
        set_drive_file_id(file_id)
    except DropboxStoreError as exc:
        raise UiStateError(f'Herladen mislukt: {exc}') from exc
    except Exception as exc:
        raise UiStateError(f'Herladen mislukt: {exc}') from exc


def clear_geocoding_cache(*, filename: str = 'capelle_addresses.json') -> None:
    """
    Clear geocoding_cache in the Drive store.

    Args:
        filename: Store filename.

    Raises:
        UiStateError: On failures.
    """
    try:
        cache, payload, file_id = load_geocoding_cache(filename=filename)
        cache.clear()
        save_geocoding_cache(
            cache,
            payload=payload,
            file_id=file_id,
            filename=filename,
        )
    except GeocodingStoreError as exc:
        raise UiStateError(f'Wissen mislukt: {exc}') from exc
    except Exception as exc:
        raise UiStateError(f'Wissen mislukt: {exc}') from exc
