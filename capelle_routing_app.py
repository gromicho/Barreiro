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

import ast
import base64
import os

import streamlit as st
from openai import OpenAI


DEFAULT_MODEL = "gpt-4.1-mini"


def require_env(name: str) -> str:
    """
    Return the value of an environment variable or raise a Streamlit error.

    Args:
        name: Environment variable name.

    Returns:
        The environment variable value.

    Raises:
        RuntimeError: If the environment variable is missing/empty.
    """
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    """
    Convert raw image bytes to a data URL suitable for OpenAI vision input.

    Args:
        image_bytes: Raw image bytes.
        mime_type: MIME type (e.g., 'image/jpeg', 'image/png').

    Returns:
        A data URL string.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def build_address_extraction_prompt() -> str:
    """
    Build a prompt that extracts only postal addresses and returns a Python list literal.

    Returns:
        Prompt string.
    """
    return (
        "From the image, extract ONLY text fragments that look like postal addresses.\n\n"
        "What counts as a postal address (some combination of):\n"
        "- Street name + house/building number\n"
        "- Apartment/unit/suite\n"
        "- Postal/ZIP code\n"
        "- City/town/locality\n"
        "- State/province/region\n"
        "- Country\n\n"
        "Rules:\n"
        "- Extract only complete or near-complete addresses.\n"
        "- Ignore names, phone numbers, email addresses, URLs, company names, headings, and any non-address text.\n"
        "- Do NOT invent or infer missing address components.\n"
        "- If something in an address is unreadable, replace only that part with \"[unclear]\".\n"
        "- If an address is split across multiple lines, merge into ONE line using commas.\n"
        "- Remove duplicates.\n\n"
        "Output format:\n"
        "- Return ONLY a valid Python list literal of strings (e.g., [\"...\"])\n"
        "- No explanations, no extra text.\n"
        "- If none found, return []."
    )


def parse_python_list_of_strings(raw: str) -> list[str]:
    """
    Parse a Python list literal and validate it is a list[str].

    Args:
        raw: Model output expected to be a Python list literal.

    Returns:
        Parsed list of strings.

    Raises:
        ValueError: If parsing fails or the value is not a list of strings.
    """
    try:
        parsed = ast.literal_eval(raw.strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Output is not a valid Python literal: {raw!r}") from exc

    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError(f"Expected list[str], got: {type(parsed).__name__}")

    # Normalize whitespace a bit (optional but helpful for OCR)
    cleaned: list[str] = []
    for item in parsed:
        s = " ".join(item.split())
        cleaned.append(s)

    # De-dupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for addr in cleaned:
        if addr not in seen:
            seen.add(addr)
            unique.append(addr)

    return unique


def extract_addresses_from_image(
    client: OpenAI,
    image_bytes: bytes,
    mime_type: str,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """
    Extract postal addresses from an image using an OpenAI vision-capable model.

    Args:
        client: Initialized OpenAI client.
        image_bytes: Raw image bytes.
        mime_type: MIME type for the image.
        model: Vision-capable model ID.

    Returns:
        List of extracted postal addresses.

    Raises:
        ValueError: If the model output is not a valid Python list of strings.
    """
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

    return parse_python_list_of_strings(response.output_text)


def main() -> None:
    """
    Streamlit app:
    - Take a photo
    - Extract addresses from the image
    - Display the resulting Python list
    """
    st.set_page_config(page_title="Address Extractor", layout="centered")
    st.title("📸 Address Extractor (OpenAI Vision)")

    try:
        api_key = require_env("OPENAI_API_KEY")
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    client = OpenAI(api_key=api_key)

    model = st.text_input("Model", value=DEFAULT_MODEL)
    photo = st.camera_input("Maak een foto van het briefje / adressenlijst")

    if photo is None:
        st.info("Maak een foto om te starten.")
        return

    image_bytes = photo.getvalue()
    mime_type = photo.type or "image/jpeg"

    with st.spinner("Adressen extraheren..."):
        try:
            addresses = extract_addresses_from_image(
                client=client,
                image_bytes=image_bytes,
                mime_type=mime_type,
                model=model,
            )
        except Exception as exc:
            st.error(f"Kon adressen niet extraheren: {exc}")
            st.stop()

    st.subheader("Gevonden adressen (Python list)")
    st.code(repr(addresses), language="python")

    st.subheader("Per regel")
    if addresses:
        st.text_area("Adressen", value="\n".join(addresses), height=240)
    else:
        st.write("Geen adressen gevonden.")


if __name__ == "__main__":
    main()


