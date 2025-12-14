"""Language choice stored in Streamlit session_state."""

import streamlit as st

STATE_LANG = 'ui_lang'


def _lang_from_locale(locale: str | None) -> str:
    """Map browser locale to a supported language code.

    Args:
        locale: Browser locale like 'nl-NL' or 'en-US'.

    Returns:
        A supported language code: 'nl', 'en', or 'pt'.
    """
    if not locale:
        return 'en'

    loc = str(locale).strip().lower()
    if loc.startswith('nl'):
        return 'nl'
    if loc.startswith('pt'):
        return 'pt'
    return 'en'


def init_language_if_missing(*, default_lang: str | None = None) -> None:
    """Initialize UI language in session_state.

    If default_lang is None, derive it from st.context.locale.

    Args:
        default_lang: Language code, or None.
    """
    if STATE_LANG in st.session_state:
        return

    if default_lang is None:
        locale = getattr(st.context, 'locale', None)
        default_lang = _lang_from_locale(locale)

    st.session_state[STATE_LANG] = str(default_lang)


def get_language(*, default_lang: str = 'en') -> str:
    """Get the current language code.

    Args:
        default_lang: Returned if missing/invalid.

    Returns:
        Language code.
    """
    value = st.session_state.get(STATE_LANG)
    if isinstance(value, str) and value:
        return value
    return str(default_lang)


def set_language(lang: str) -> None:
    """Set the current language code.

    Args:
        lang: Language code.
    """
    st.session_state[STATE_LANG] = str(lang)
