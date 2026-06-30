"""OpenAI vision routing for generated-image prompts."""

import base64
import json

from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import settings

_openai = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None


class ProductPromptRoute(BaseModel):
    """Small classification used only to choose product prompt packs."""

    product_type: str = Field(description="One of: bag, perfume, glasses, other")
    product_kind: str = Field(description="Specific kind within the product type")
    confidence: float = Field(ge=0, le=1)
    reason: str


def _image_url(image_bytes: bytes, media_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"


def classify_for_prompt(image_bytes: bytes, media_type: str) -> ProductPromptRoute:
    """Classify the product with OpenAI vision before prompt selection."""
    if not _openai:
        return ProductPromptRoute(
            product_type="bag",
            product_kind="handbag",
            confidence=0,
            reason="OPENAI_API_KEY is not configured; defaulted to bag/handbag prompt.",
        )

    response = _openai.chat.completions.create(
        model=settings.openai_vision_model,
        temperature=0,
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the product image for ecommerce prompt routing. "
                    "Return only JSON with product_type, product_kind, confidence, and reason. "
                    "product_type must be one of: bag, perfume, glasses, other. "
                    "For bag product_kind use one of: purse, handbag, tote, suitcase, backpack, other. "
                    "Use purse for clutches, minaudieres, wristlets, small evening bags, "
                    "and compact hand-carried bags. Use tote for large open carryall bags. "
                    "Use suitcase for luggage/travel cases. For perfume use kinds like "
                    "eau de parfum, perfume oil, body mist, gift set, bottle, boxed bottle. "
                    "For glasses use kinds like sunglasses, eyeglasses, readers, frames. "
                    "If unsure, choose the closest product_type and set confidence below 0.6."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Classify this product for image-generation prompt routing.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_url(image_bytes, media_type)},
                    },
                ],
            },
        ],
    )

    text = response.choices[0].message.content or "{}"
    data = json.loads(text)
    route = ProductPromptRoute.model_validate(data)
    if route.product_type not in {"bag", "perfume", "glasses", "other"}:
        route.product_type = "bag"
    if route.product_type == "bag" and route.product_kind not in {
        "purse",
        "handbag",
        "tote",
        "suitcase",
        "backpack",
        "other",
    }:
        route.product_kind = "handbag"
    return route
