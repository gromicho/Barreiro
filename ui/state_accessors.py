"""
State accessors and normalization helpers.
"""

import streamlit as st

from ui.state_keys import (
    STATE_ADDRESSES_TEXT,
    STATE_DRIVE_FILE_ID,
    STATE_DRIVE_PAYLOAD,
    STATE_STORE_FILENAME,
)


def normalize_addresses_text(text: str) -> str:
    """
    Normalize address list text: trim lines, drop empties, join with \\n, end with \\n.

    Args:
        text: Raw text (one address per line).

    Returns:
        Normalized text (possibly empty).
    """
    lines = [ln.strip() for ln in str(text).splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ''
    return '\n'.join(lines) + '\n'


def get_store_filename(*, default: str = 'capelle_addresses.json') -> str:
    """
    Get the Drive store filename from session_state.

    Args:
        default: Default filename if missing.

    Returns:
        Store filename.
    """
    value = st.session_state.get(STATE_STORE_FILENAME)
    return str(value) if isinstance(value, str) and value else str(default)


def set_store_filename(filename: str) -> None:
    """
    Set the Drive store filename in session_state.

    Args:
        filename: Store filename.
    """
    st.session_state[STATE_STORE_FILENAME] = str(filename)


def get_addresses_text() -> str:
    """
    Get the current addresses text from session_state.

    Returns:
        Addresses text (possibly empty).
    """
    return str(st.session_state.get(STATE_ADDRESSES_TEXT, '') or '')


def set_addresses_text(value: str) -> None:
    """
    Set the addresses text in session_state.

    Args:
        value: New text.
    """
    st.session_state[STATE_ADDRESSES_TEXT] = str(value)


def get_drive_payload() -> dict[str, object]:
    """
    Get the Drive payload dict from session_state.

    Returns:
        Payload dict (empty dict if missing/invalid).
    """
    payload = st.session_state.get(STATE_DRIVE_PAYLOAD)
    return payload if isinstance(payload, dict) else {}


def set_drive_payload(payload: dict[str, object]) -> None:
    """
    Set the Drive payload dict in session_state.

    Args:
        payload: Payload dict.
    """
    st.session_state[STATE_DRIVE_PAYLOAD] = dict(payload)


def get_drive_file_id() -> str | None:
    """
    Get the Drive file id from session_state.

    Returns:
        File id, or None.
    """
    value = st.session_state.get(STATE_DRIVE_FILE_ID)
    return str(value) if isinstance(value, str) else None


def set_drive_file_id(file_id: str | None) -> None:
    """
    Set the Drive file id in session_state.

    Args:
        file_id: Drive file id or None.
    """
    st.session_state[STATE_DRIVE_FILE_ID] = file_id
