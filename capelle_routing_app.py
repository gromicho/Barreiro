# from __future__ import annotations

# from routing.app import RoutingAppConfig, run_routing_app


# def main() -> None:
#     """Capelle entry point."""
#     cfg = RoutingAppConfig(
#         store_filename='capelle_addresses.json',
#         drive_prefix='capelle_drive',
#         title_name='Joaquim Gromicho',
#         title_city='Capelle aan den IJssel',
#     )
#     run_routing_app(cfg=cfg)


# if __name__ == '__main__':
#     main()

from __future__ import annotations

import base64
import json
import os
import re

import streamlit as st
from openai import OpenAI


DEFAULT_MODEL = "gpt-4.1-mini"


def require_env(name: str) -> str:
    """
    Read an environment variable or raise a Streamlit-friendly error.

    Args:
        name: Environment variable name.

    Returns:
        The environment variable value.

    Raises:
        RuntimeError: If missing or empty.
    """
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    """
    Convert raw image bytes to a data URL for OpenAI vision input.

    Args:
        image_bytes: Raw bytes of the image.
        mime_type: MIME type (e.g. 'image/jpeg', 'image/png').

    Returns:
        Data URL suitable for OpenAI vision input.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def build_address_extraction_prompt() -> str:
    """
    Build a strict prompt for extracting addresses.

    Uses JSON output for robustness (easier to parse than a Python literal).

    Returns:
        Prompt string.
    """
    return (
        "Extract ONLY postal addresses from the image.\n\n"
        "A postal address typically contains some combination of:\n"
        "- Street name + house/building number\n"
        "- Apartment/unit/suite (optional)\n"
        "- Postal/ZIP code\n"
        "- City/town/locality\n"
        "- State/province/region (optional)\n"
        "- Country (optional)\n\n"
        "Rules:\n"
        "- Extract only complete or near-complete addresses.\n"
        "- Ignore names, phone numbers, emails, URLs, company names, headings, and any non-address text.\n"
        "- Do NOT invent or infer missing components.\n"
        "- If part of an address is unreadable, replace only that part with \"[unclear]\".\n"
        "- If an address spans multiple lines, merge into ONE line using commas.\n"
        "- Remove duplicates.\n\n"
        "Output:\n"
        "- Return ONLY valid JSON.\n"
        "- The JSON must be an array of strings.\n"
        "- Example: [\"Street 1, 1234 AB City\", \"Other Rd 9, 99999 Town\"]\n"
        "- If none found, return []\n"
    )


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """
    De-duplicate strings while preserving their original order.

    Args:
        items: List of strings.

    Returns:
        De-duplicated list.
    """
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def normalize_addresses(addresses: list[str]) -> list[str]:
    """
    Normalize whitespace and basic punctuation spacing in extracted addresses.

    Args:
        addresses: Extracted addresses.

    Returns:
        Cleaned addresses (still faithful to the source).
    """
    cleaned: list[str] = []
    for a in addresses:
        # Collapse whitespace
        s = " ".join(a.split())

        # Light cleanup around commas
        s = re.sub(r"\s*,\s*", ", ", s).strip()

        cleaned.append(s)

    return _dedupe_preserve_order(cleaned)


def parse_json_list_of_strings(raw: str) -> list[str]:
    """
    Parse model output as JSON array of strings.

    Also attempts a small repair if the model wraps JSON in extra text.

    Args:
        raw: Model output expected to be JSON.

    Returns:
        Parsed list[str].

    Raises:
        ValueError: If parsing fails or isn't list[str].
    """
    text = raw.strip()

    # First try: strict JSON
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Repair attempt: extract first JSON array from the text
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise ValueError(f"Output is not valid JSON and no JSON array found: {raw!r}")
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError(f"Expected JSON array of strings, got: {type(parsed).__name__}")

    return normalize_addresses(parsed)


@st.cache_resource
def get_client() -> OpenAI:
    """
    Create and cache the OpenAI client (Streamlit reruns frequently).

    Returns:
        Cached OpenAI client.
    """
    api_key = require_env("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


@st.cache_data(show_spinner=False)
def extract_addresses_cached(
    image_bytes: bytes,
    mime_type: str,
    model: str,
) -> tuple[list[str], str]:
    """
    Extract addresses from an image (cached by Streamlit).

    Args:
        image_bytes: Raw image bytes.
        mime_type: Image MIME type.
        model: Model ID.

    Returns:
        (addresses, raw_model_output)
    """
    client = get_client()
    data_url = image_bytes_to_data_url(image_bytes, mime_type)
    prompt = build_address_extraction_prompt()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    raw = response.output_text
    addresses = parse_json_list_of_strings(raw)
    return addresses, raw


def main() -> None:
    """
    Streamlit app: camera input -> address extraction -> display list.
    """
    st.set_page_config(page_title="Address Extractor", layout="centered")
    st.title("📸 Address Extractor (OpenAI Vision)")

    # Early config / validation
    try:
        _ = require_env("OPENAI_API_KEY")
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        model = st.text_input("Model", value=DEFAULT_MODEL)
    with col2:
        show_debug = st.checkbox("Show debug", value=False)

    photo = st.camera_input("Maak een foto van het briefje / adressenlijst")
    if photo is None:
        st.info("Maak een foto om te starten.")
        return

    image_bytes = photo.getvalue()
    mime_type = photo.type or "image/jpeg"

    with st.spinner("Adressen extraheren..."):
        try:
            addresses, raw = extract_addresses_cached(
                image_bytes=image_bytes,
                mime_type=mime_type,
                model=model,
            )
        except Exception as exc:
            st.error(f"Kon adressen niet extraheren: {exc}")
            if show_debug:
                st.subheader("Raw model output")
                st.code(getattr(exc, "args", [""])[0] if exc.args else "", language="text")
            st.stop()

    st.subheader("Gevonden adressen (Python list)")
    st.code(repr(addresses), language="python")

    st.subheader("Per regel")
    if addresses:
        st.text_area("Adressen", value="\n".join(addresses), height=240)
    else:
        st.write("Geen adressen gevonden.")

    if show_debug:
        st.subheader("Raw model output")
        st.code(raw, language="json")


if __name__ == "__main__":
    main()
    