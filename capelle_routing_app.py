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

import base64
import os

import streamlit as st
from openai import OpenAI


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    """Convert raw image bytes to a data URL suitable for OpenAI vision input."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def extract_text_from_image(client: OpenAI, image_bytes: bytes, mime_type: str) -> str:
    """Extract text from an image using an OpenAI vision-capable model."""
    data_url = image_bytes_to_data_url(image_bytes, mime_type)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Extract all text from this image. "
                            "Return only the text, preserving line breaks. "
                            "If something is unclear, mark it as [unclear]."
                        ),
                    },
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    return response.output_text


def main() -> None:
    """Streamlit app: capture photo, send to OpenAI, show extracted text."""
    st.title("Camera -> OCR (OpenAI vision)")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.warning("Set OPENAI_API_KEY as an environment variable.")
        st.stop()

    client = OpenAI(api_key=api_key)

    photo = st.camera_input("Maak een foto van het briefje / adressenlijst")
    if photo is None:
        return

    # Streamlit provides an UploadedFile-like object
    image_bytes = photo.getvalue()
    mime_type = photo.type or "image/jpeg"

    with st.spinner("Tekst extracten..."):
        text = extract_text_from_image(client, image_bytes, mime_type)

    st.subheader("Gevonden tekst")
    st.text_area("Output", value=text, height=300)


if __name__ == "__main__":
    main()

