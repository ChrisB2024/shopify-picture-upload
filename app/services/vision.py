"""Claude calls: classify product, write listing, and QC generated images."""

import base64
import json
import re
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from app.config import settings
from app.product_types.prompts import ProductPromptPack
from app.schemas import ImageCandidateReview, Listing, ProductAnalysis

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _image_block(image_bytes: bytes, media_type: str) -> dict:
    """Build a base64 image content block for the Messages API."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,  # "image/jpeg" or "image/png"
            "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
        },
    }


def _response_text(response) -> str:
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _json_prompt(prompt: str, schema: type[BaseModel]) -> str:
    return (
        f"{prompt}\n\n"
        "Return only valid JSON. Do not wrap it in Markdown. Do not include any "
        "introductory or trailing text. The JSON must match this schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )


def _extract_json(text: str) -> str:
    """Handle plain JSON and the common ```json fenced response."""
    stripped = text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    start = min(
        [idx for idx in (stripped.find("{"), stripped.find("[")) if idx != -1],
        default=-1,
    )
    if start == -1:
        return stripped

    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end <= start:
        return stripped
    return stripped[start : end + 1]


def _create_parsed(
    *,
    model: str,
    max_tokens: int,
    content: list[dict],
    output_schema: type[SchemaT],
) -> SchemaT:
    """Anthropic SDK 0.25-compatible structured output helper."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    text = _response_text(response)
    json_text = _extract_json(text)
    try:
        return output_schema.model_validate_json(json_text)
    except Exception as exc:
        raise RuntimeError(
            f"Claude returned invalid JSON for {output_schema.__name__}: {text}"
        ) from exc


def classify_product(
    image_bytes: bytes,
    media_type: str,
    prompt_pack: ProductPromptPack,
    product_kind: str,
) -> ProductAnalysis:
    """Cheap pass: look at the photo, return structured attributes.

    The installed Anthropic SDK exposes messages.create, so we validate the JSON
    response against the Pydantic model locally.
    """
    prompt = (
        f"{prompt_pack.classify_prompt}\n\n"
        f"Product type route: {prompt_pack.product_type}\n"
        f"Product kind route: {product_kind}\n"
        "Use those routes unless the image clearly contradicts them."
    )

    return _create_parsed(
        model=settings.classify_model,
        max_tokens=1024,
        content=[
            _image_block(image_bytes, media_type),
            {"type": "text", "text": _json_prompt(prompt, ProductAnalysis)},
        ],
        output_schema=ProductAnalysis,
    )


def classify_bag(image_bytes: bytes, media_type: str) -> ProductAnalysis:
    """Backward-compatible bag classifier."""
    from app.product_types.prompts import BAG_PROMPT_PACK

    return classify_product(image_bytes, media_type, BAG_PROMPT_PACK, "handbag")


def write_listing(
    analysis: ProductAnalysis,
    image_bytes: bytes,
    media_type: str,
    prompt_pack: ProductPromptPack,
) -> Listing:
    """Quality pass: write the listing copy in OHH Bags' voice.

    Passes both the structured analysis AND the image so Opus can ground the
    copy in what it actually sees.
    """
    prompt = (
        f"{prompt_pack.listing_prompt}\n\n"
        "Brand/store voice: polished, concise, ecommerce-ready, warm but not hypey. "
        "Do not invent unverified claims, sizes, ingredients, scent notes, UV ratings, "
        "designer names, or materials that are not visible or provided. "
        "Use clean Shopify HTML for description_html.\n\n"
        f"Detected attributes: {analysis.model_dump_json(indent=2)}"
    )

    return _create_parsed(
        model=settings.copy_model,
        max_tokens=2048,
        content=[
            _image_block(image_bytes, media_type),
            {"type": "text", "text": _json_prompt(prompt, Listing)},
        ],
        output_schema=Listing,
    )


def review_image_candidate(
    original_bytes: bytes,
    candidate_bytes: bytes,
    media_type: str,
    candidate_path: str,
    kind: str,
    prompt_pack: ProductPromptPack,
) -> ImageCandidateReview:
    """Rank a generated image candidate before human review.

    This is an automated triage pass only. It can make review faster, but it
    should not be treated as publication approval.
    """
    hands_instruction = (
        "Also check whether the hands look anatomically plausible and whether "
        "they avoid covering the product's front, label, clasp, hardware, frame, "
        "or other distinctive details."
        if kind == "on_model"
        else "Set hands_ok to null because there should be no hands in this image."
    )
    prompt = (
        "Compare the original product photo to the generated candidate. "
        "Score only ecommerce usefulness and product fidelity, not artistic taste. "
        f"The most important checks are {prompt_pack.review_focus}. "
        "If the generated product changes the shape, outline, aspect ratio, side "
        "profile, key detail locations, or material layout, set silhouette_ok or "
        "proportions_ok to false, set publishable to false, and cap score at 2 "
        "even if the image looks attractive. Penalize "
        "any color shift, changed material, warped silhouette, missing or invented "
        "logos/text/hardware/detailing, hidden product, or unrealistic placement. "
        f"{hands_instruction}\n\n"
        f"Candidate kind: {kind}\n"
        f"Candidate path: {candidate_path}"
    )

    review = _create_parsed(
        model=settings.classify_model,
        max_tokens=1024,
        content=[
            {"type": "text", "text": "Original product photo:"},
            _image_block(original_bytes, media_type),
            {"type": "text", "text": "Generated candidate:"},
            _image_block(candidate_bytes, "image/png"),
            {"type": "text", "text": _json_prompt(prompt, ImageCandidateReview)},
        ],
        output_schema=ImageCandidateReview,
    )
    review.path = candidate_path
    review.kind = kind
    if kind != "on_model":
        review.hands_ok = None
    return review
