from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str

    # Store/brand this run is generating for. Drives listing voice + SEO suffix,
    # so a new client is just a .env change — no code edits. Override per store.
    brand_name: str = "OHH Bags"

    photoroom_api_key: str = ""
    pixelcut_api_key: str = ""
    # Background removal backend:
    # - photoroom: paid API, strongest/most consistent
    # - pixelcut: paid API, returns transparent PNG cutouts for local compositing
    # - rembg: local/free after setup, cheaper but needs visual QA
    background_remover: str = "photoroom"

    openai_api_key: str = ""
    openai_vision_model: str = "gpt-4o-mini"
    # OpenAI image EDITS model (takes the product photo as input and modifies
    # from it — not fresh generation). Keep configurable; image models move fast.
    # gpt-image-2 processes every input at high fidelity automatically, so
    # product fidelity (non-negotiable for us) needs no tuning.
    image_model: str = "gpt-image-2"
    lifestyle_image_size: str = "1024x1024"
    on_model_image_size: str = "1024x1536"
    image_quality: str = "high"
    # gpt-image-1 ONLY. gpt-image-2 auto-processes inputs at high fidelity and
    # REJECTS this param — the images.py call only sends it for "gpt-image-1",
    # so it's a no-op on gpt-image-2 (kept for easy fallback to gpt-image-1).
    image_input_fidelity: str = "high"
    lifestyle_candidates: int = 3
    # How many on-model candidates to generate per bag (you cherry-pick).
    on_model_candidates: int = 3

    shopify_store: str = ""
    # Dev Dashboard apps use the client-credentials grant: exchange these for a
    # 24h Admin API token at runtime (see services/shopify.py). Same-org only.
    shopify_client_id: str = ""
    shopify_client_secret: str = ""
    shopify_api_version: str = "2024-10"

    # Model choices — Haiku for the cheap classify pass, Opus for the copy.
    classify_model: str = "claude-haiku-4-5"
    copy_model: str = "claude-opus-4-8"

    # Where generated images land for review (single-user local tool).
    output_dir: str = "outputs"

    # Approved Shopify copy examples used as style-only few-shot context.
    listing_examples_path: str = "outputs/shopify_examples/listing_examples.jsonl"
    listing_example_count: int = 3

    # Approved store IMAGES, tagged by tier, used as visual style references for
    # generation. Built by `python -m app.services.image_examples`.
    image_examples_path: str = "outputs/shopify_examples/image_examples.jsonl"


settings = Settings()
