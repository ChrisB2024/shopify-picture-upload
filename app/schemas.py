"""Typed contracts passed between pipeline steps.

These Pydantic models double as the structured-output schemas for the Claude
calls (see services/vision.py — responses are validated against them).
"""

from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


class ProductAnalysis(BaseModel):
    """Output of the cheap classify pass — what is this product."""

    product_type: str = Field(description="e.g. bag, perfume, glasses")
    product_kind: str = Field(description="e.g. clutch, eau de parfum, sunglasses")
    primary_color: str
    material: str = Field(description="e.g. leather, glass, acetate, metal")
    style_keywords: list[str] = Field(description="3-6 short descriptive tags")
    notable_features: list[str] = Field(
        default_factory=list,
        description="Product-specific details visible in the reference photos",
    )


class BagAnalysis(ProductAnalysis):
    """Backward-compatible name for older bag-specific imports."""


class Listing(BaseModel):
    """Output of the Opus copy pass — the actual Shopify listing fields."""

    title: str = Field(description="Product title, in the store's brand voice")
    description_html: str = Field(description="Body HTML for the product page")
    seo_title: str = Field(max_length=70)
    seo_description: str = Field(max_length=160)
    tags: list[str]

    @field_validator("seo_title", mode="before")
    @classmethod
    def trim_seo_title(cls, value: str) -> str:
        text = str(value).strip()
        return text[:70].rstrip()

    @field_validator("seo_description", mode="before")
    @classmethod
    def trim_seo_description(cls, value: str) -> str:
        text = str(value).strip()
        return text[:160].rstrip()


class ListingGenerationContext(BaseModel):
    """Validated current-product data supplied to grouped listing generation."""

    product_family_name: str = Field(min_length=1, max_length=200)
    variant_names: list[
        Annotated[str, Field(min_length=1, max_length=100)]
    ] = Field(min_length=1, max_length=100)
    variant_count: int = Field(ge=1, le=2048)

    @model_validator(mode="after")
    def validate_variant_count(self) -> "ListingGenerationContext":
        if self.variant_count < len(self.variant_names):
            raise ValueError("variant_count cannot be smaller than variant_names")
        return self


class ImageCandidateReview(BaseModel):
    """Vision QC notes for a generated candidate before human review."""

    path: str
    kind: str = Field(description="lifestyle or on_model")
    score: int = Field(ge=1, le=5, description="5 means strongest candidate")
    product_match: int = Field(ge=1, le=5)
    silhouette_ok: bool
    proportions_ok: bool
    product_visible: bool
    hardware_ok: bool
    hands_ok: bool | None = Field(
        default=None, description="Only relevant for on-model candidates"
    )
    publishable: bool
    notes: str


class ProductPromptRoute(BaseModel):
    """OpenAI vision result used to choose generated-image prompts."""

    product_type: str = Field(description="bag, perfume, glasses, or other")
    product_kind: str
    confidence: float = Field(ge=0, le=1)
    reason: str


class ProcessResult(BaseModel):
    """What the API returns to the caller after the full pipeline runs."""

    analysis: ProductAnalysis
    listing: Listing
    prompt_route: ProductPromptRoute | None = None

    # Guaranteed: white-bg ecommerce shot (real product pixels).
    white_bg_path: str | None = None

    # Best-effort: generated candidates saved for human review. The pipeline
    # never auto-publishes these — you pick the good ones.
    lifestyle_candidate_paths: list[str] = Field(default_factory=list)
    on_model_candidate_paths: list[str] = Field(default_factory=list)
    candidate_reviews: list[ImageCandidateReview] = Field(default_factory=list)
    review_required: bool = True
    image_errors: list[str] = Field(default_factory=list)

    shopify_product_id: str | None = None
    shopify_admin_url: str | None = None
