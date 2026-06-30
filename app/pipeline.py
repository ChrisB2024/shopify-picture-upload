"""The pipeline: phone photo -> analysis -> copy -> images -> Shopify draft.

Thin orchestration. Each step lives in a service module. Image generation has
three tiers: a guaranteed white-bg shot, simpler lifestyle candidates, and
best-effort on-model candidates saved for human review (never auto-published).
"""

import uuid
from pathlib import Path

from app.config import settings
from app.product_types.prompts import get_prompt_pack
from app.schemas import ImageCandidateReview, ProcessResult, ProductPromptRoute
from app.services import images, product_router, shopify, vision


def _save(image_bytes: bytes, run_dir: Path, name: str) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    path.write_bytes(image_bytes)
    return str(path)


async def run(
    image_bytes: bytes,
    media_type: str,
    reference_images: list[tuple[bytes, str]] | None = None,
    create_shopify: bool = True,
    creative_mode: str = "none",
    qc_enabled: bool = False,
    lifestyle_count: int | None = None,
    on_model_count: int | None = None,
) -> ProcessResult:
    run_dir = Path(settings.output_dir) / uuid.uuid4().hex[:8]
    reference_images = reference_images or [(image_bytes, media_type)]
    for i, (reference_bytes, reference_media_type) in enumerate(reference_images):
        ext = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
        }.get(reference_media_type, "bin")
        name = "original" if i == 0 else f"original_{i}"
        _save(reference_bytes, run_dir, f"{name}.{ext}")
    image_errors: list[str] = []

    prompt_route: ProductPromptRoute | None = None
    product_type = "bag"
    product_kind = "handbag"
    try:
        route = product_router.classify_for_prompt(image_bytes, media_type)
        prompt_route = ProductPromptRoute.model_validate(route.model_dump())
        product_type = prompt_route.product_type
        product_kind = prompt_route.product_kind
    except Exception as exc:
        image_errors.append(f"prompt_route: {exc}")
    prompt_pack = get_prompt_pack(product_type)

    # 1. Cheap classify pass (Haiku) — what is this product.
    analysis = vision.classify_product(
        image_bytes,
        media_type,
        prompt_pack,
        product_kind,
    )

    # 2. Quality copy pass (Opus) — listing in OHH voice.
    listing = vision.write_listing(analysis, image_bytes, media_type, prompt_pack)

    # 3a. Guaranteed white-bg ecommerce shot (Photoroom).
    white_bg_path: str | None = None
    generation_references = list(reference_images)
    try:
        white_bg = await images.clean_white_bg(image_bytes, media_type)
        white_bg_path = _save(white_bg, run_dir, "white_bg.png")
        generation_references = [(white_bg, "image/png")] + reference_images[1:]
    except Exception as exc:
        image_errors.append(f"white_bg: {exc}")

    # 3b. Best-effort lifestyle candidates. Use the clean product reference
    # when available; it gives the image model a clearer shape to preserve.
    lifestyle_paths: list[str] = []
    lifestyle_blobs: list[tuple[str, bytes]] = []
    if creative_mode in {"lifestyle", "both"}:
        try:
            lifestyle_candidates = await images.generate_lifestyle(
                generation_references,
                prompt_pack,
                count=lifestyle_count,
            )
            for i, candidate in enumerate(lifestyle_candidates):
                path = _save(candidate, run_dir, f"lifestyle_{i}.png")
                lifestyle_paths.append(path)
                lifestyle_blobs.append((path, candidate))
        except Exception as exc:
            image_errors.append(f"lifestyle: {exc}")

    # 3c. Best-effort on-model candidates. NEVER blocks the draft — a failure
    # here just means no model shots this run; the white-bg + listing still go.
    on_model_paths: list[str] = []
    on_model_blobs: list[tuple[str, bytes]] = []
    if creative_mode in {"model", "both"}:
        try:
            candidates = await images.generate_on_model(
                generation_references,
                product_kind=product_kind,
                prompt_pack=prompt_pack,
                count=on_model_count,
            )
            for i, candidate in enumerate(candidates):
                path = _save(candidate, run_dir, f"on_model_{i}.png")
                on_model_paths.append(path)
                on_model_blobs.append((path, candidate))
        except Exception as exc:
            image_errors.append(f"on_model: {exc}")

    # 3d. Automated QC triage for review. Failures here do not throw away the
    # generated images; the user can still inspect the files manually.
    candidate_reviews: list[ImageCandidateReview] = []
    if qc_enabled:
        for kind, blobs in (
            ("lifestyle", lifestyle_blobs),
            ("on_model", on_model_blobs),
        ):
            for path, candidate in blobs:
                try:
                    candidate_reviews.append(
                        vision.review_image_candidate(
                            image_bytes,
                            candidate,
                            media_type,
                            path,
                            kind,
                            prompt_pack,
                        )
                    )
                except Exception as exc:
                    image_errors.append(f"review {path}: {exc}")

    candidate_reviews.sort(
        key=lambda review: (
            review.publishable,
            review.silhouette_ok,
            review.proportions_ok,
            review.score,
            review.product_match,
            review.product_visible,
        ),
        reverse=True,
    )

    generated_images_by_path = dict(lifestyle_blobs + on_model_blobs)
    shopify_image_bytes = [
        generated_images_by_path[review.path]
        for review in candidate_reviews
        if (
            review.publishable
            and review.silhouette_ok
            and review.proportions_ok
            and review.path in generated_images_by_path
        )
    ]
    if not shopify_image_bytes and generated_images_by_path:
        shopify_image_bytes = list(generated_images_by_path.values())
    if not shopify_image_bytes and white_bg_path:
        shopify_image_bytes = [Path(white_bg_path).read_bytes()]

    # 4. Create the Shopify DRAFT (never publishes). Attach generated product
    # images only. original.jpg and white_bg.png are intentionally excluded.
    product_id = None
    admin_url = None
    if not create_shopify:
        image_errors.append("shopify: skipped by caller")
    elif shopify_image_bytes:
        try:
            result = await shopify.create_draft(listing, shopify_image_bytes)
            product_id, admin_url = result["product_id"], result["admin_url"]
            image_errors.extend(f"shopify: {error}" for error in result["image_errors"])
        except Exception as exc:
            image_errors.append(f"shopify: {exc}")
    else:
        image_errors.append(
            "shopify: skipped because no generated publishable images were available"
        )

    return ProcessResult(
        analysis=analysis,
        listing=listing,
        prompt_route=prompt_route,
        white_bg_path=white_bg_path,
        lifestyle_candidate_paths=lifestyle_paths,
        on_model_candidate_paths=on_model_paths,
        candidate_reviews=candidate_reviews,
        review_required=True,
        image_errors=image_errors,
        shopify_product_id=product_id,
        shopify_admin_url=admin_url,
    )
