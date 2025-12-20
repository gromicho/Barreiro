"""
Streamlit session_state keys and initialization.
"""

import streamlit as st

STATE_ADDRESSES_TEXT = 'addresses_text'
STATE_DRIVE_FILE_ID = 'drive_file_id'
STATE_DRIVE_PAYLOAD = 'drive_payload'
STATE_STORE_FILENAME = 'store_filename'


def init_state_if_missing(*, filename: str = 'capelle') -> None:
    """
    Initialize Streamlit session_state keys if missing.

    Args:
        filename: Instance name (e.g. 'capelle').
    """
    if STATE_STORE_FILENAME not in st.session_state:
        st.session_state[STATE_STORE_FILENAME] = str(filename)

    if STATE_ADDRESSES_TEXT not in st.session_state:
        st.session_state[STATE_ADDRESSES_TEXT] = ''

    if STATE_DRIVE_PAYLOAD not in st.session_state:
        st.session_state[STATE_DRIVE_PAYLOAD] = {}

    if STATE_DRIVE_FILE_ID not in st.session_state:
        st.session_state[STATE_DRIVE_FILE_ID] = None
