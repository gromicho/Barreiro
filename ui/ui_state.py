"""Streamlit UI state helpers (facade).

This module preserves the original ui.ui_state import surface, but the actual
implementation is split across smaller modules in the ui package.
"""

from ui.errors import UiStateError
from ui.state_keys import (
    STATE_ADDRESSES_TEXT,
    STATE_DRIVE_FILE_ID,
    STATE_DRIVE_PAYLOAD,
    STATE_STORE_FILENAME,
    init_state_if_missing,
)
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
from ui.drive_handlers import ensure_addresses_loaded
from ui.widgets import (
    addresses_text_area,
    clear_geocoding_cache_button,
    drive_buttons_row,
    drive_version_loader,
)

__all__ = [
    'UiStateError',
    'STATE_ADDRESSES_TEXT',
    'STATE_DRIVE_FILE_ID',
    'STATE_DRIVE_PAYLOAD',
    'STATE_STORE_FILENAME',
    'normalize_addresses_text',
    'init_state_if_missing',
    'get_store_filename',
    'get_addresses_text',
    'set_addresses_text',
    'get_drive_payload',
    'set_drive_payload',
    'get_drive_file_id',
    'set_drive_file_id',
    'ensure_addresses_loaded',
    'addresses_text_area',
    'drive_buttons_row',
    'drive_version_loader',
    'clear_geocoding_cache_button',
]
