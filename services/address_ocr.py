from __future__ import annotations

import base64
import json
import re

from openai import OpenAI


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    """Convert image bytes to a data URL for OpenAI vision input."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def build_address_prompt() -> str:
    """Strict prompt: extract only addresses, return JSON list[str]."""
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
        "- If part is unreadable, replace only that part with \"[unclear]\".\n"
        "- Merge multi-line addresses into ONE line using commas.\n"
        "- Remove duplicates.\n\n"
        "Output:\n"
        "- Return ONLY valid JSON: an array of strings.\n"
        "- If none found, return []."
    )


def parse_json_list_of_strings(raw: str) -> list[str]:
    """Parse JSON array of strings; repair if model wraps the array in text."""
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise ValueError(f"No JSON array found in model output: {raw!r}")
        data = json.loads(match.group(0))

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("Expected a JSON array of strings.")

    # Normalize whitespace & de-dupe (preserve order)
    out: list[str] = []
    seen: set[str] = set()
    for s in data:
        s2 = re.sub(r"\s+", " ", s).strip()
        s2 = re.sub(r"\s*,\s*", ", ", s2)
        if s2 and s2 not in seen:
            seen.add(s2)
            out.append(s2)

    return out


def extract_addresses_from_image(
    *,
    client: OpenAI,
    image_bytes: bytes,
    mime_type: str,
    model: str,
) -> tuple[list[str], str]:
    """
    Extract addresses from an image.

    Returns:
        (addresses, raw_model_output)
    """
    data_url = image_bytes_to_data_url(image_bytes, mime_type)
    prompt = build_address_prompt()

    resp = client.responses.create(
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

    raw = resp.output_text
    return parse_json_list_of_strings(raw), raw
