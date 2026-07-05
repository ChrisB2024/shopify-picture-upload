from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str

    photoroom_api_key: str = ""
    pixelcut_api_key: str = ""
    # Background removal backend:
    # - photoroom: paid API, strongest/most consistent
    # - pixelcut: paid API, multipart upload with product cutout output
    # - rembg: local/free after setup, cheaper but needs visual QA
    background_remover: str = "photoroom"

    openai_api_key: str = ""
    openai_vision_model: str = "gpt-4o-mini"
    # OpenAI image edits model. Keep configurable because image models move fast.
    # Swap this + the impl in services/images.py to move to Gemini, etc.
    image_model: str = "gpt-image-1"
    lifestyle_image_size: str = "1024x1024"
    on_model_image_size: str = "1024x1536"
    image_quality: str = "high"
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


settings = Settings()
