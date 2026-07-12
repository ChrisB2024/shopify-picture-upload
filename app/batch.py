"""Batch runner for ready incoming product folders.

Usage:
    python -m app.batch --limit 2
    python -m app.batch --creative none --limit 1
    python -m app.batch --mode leaf --creative both --limit 2
    python -m app.batch --all

By default the runner processes product family folders. Child leaf folders are
treated as variants on one Shopify product. Use --mode leaf only when every leaf
folder should intentionally become its own product. The default grouped mode
uses white-background product images only. Creative images are opt-in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app import pipeline
from app.config import settings
from app.product_types.prompts import get_prompt_pack
from app.schemas import Listing, ListingGenerationContext
from app.services import image_examples, images, shopify, vision

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_ROOTS = ("incoming/bags", "incoming/perfumes", "incoming/glasses")
IMAGE_ORDER = {
    "front": 0,
    "side": 1,
    "angle": 2,
    "back": 3,
    "detail-1": 4,
    "detail_1": 4,
    "detail1": 4,
    "detail-2": 5,
    "detail_2": 5,
    "detail2": 5,
}
# Semantic studio-backdrop slots the router chooses between. Each store supplies
# its own PNG per slot as <settings.backdrops_dir>/<slot>.png (gitignored local
# assets), so slots resolve by convention instead of hardcoded store filenames.
BACKDROP_SLOTS = (
    "default",
    "cool_gray",
    "beige_plaster",
    "clean_white",
    "charcoal",
    "terracotta",
)


def _backdrop_path(slot: str) -> Path:
    """Resolve a semantic backdrop slot to this store's local PNG."""
    return Path(settings.backdrops_dir) / f"{slot}.png"
LIGHT_OR_METALLIC_TERMS = {
    "white",
    "cream",
    "clear",
    "acrylic",
    "crystal",
    "rhinestone",
    "rhinestones",
    "pearl",
    "bead",
    "beaded",
    "silver",
}
WARM_COLOR_TERMS = {
    "brown",
    "tan",
    "camel",
    "taupe",
    "orange",
    "yellow",
    "mustard",
    "red",
    "pink",
    "rose",
    "burgundy",
}
PRINT_TERMS = {
    "print",
    "floral",
    "geometric",
    "kente",
    "wax",
    "mosaic",
    "multicolor",
    "monogram",
    "graffiti",
    "leopard",
    "python",
    "croc",
    "alligator",
}


def _image_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem.lower()
    return (IMAGE_ORDER.get(stem, 100), stem)


def _image_files(folder: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=_image_sort_key,
    )


def _ready_product_folders(roots: list[Path]) -> list[tuple[Path, list[Path]]]:
    products: list[tuple[Path, list[Path]]] = []
    for root in roots:
        if not root.exists():
            continue
        for folder in sorted(path for path in root.rglob("*") if path.is_dir()):
            files = _image_files(folder)
            subdirs = [path for path in folder.iterdir() if path.is_dir()]
            if files and not subdirs:
                products.append((folder, files))
    return products


def _variant_groups(roots: list[Path]) -> list[tuple[Path, list[tuple[str, Path, list[Path]]]]]:
    groups: list[tuple[Path, list[tuple[str, Path, list[Path]]]]] = []
    for root in roots:
        if not root.exists():
            continue
        for product_folder in sorted(path for path in root.iterdir() if path.is_dir()):
            variants: list[tuple[str, Path, list[Path]]] = []
            for folder in sorted(path for path in product_folder.rglob("*") if path.is_dir()):
                files = _image_files(folder)
                subdirs = [path for path in folder.iterdir() if path.is_dir()]
                if files and not subdirs:
                    variant_name = str(folder.relative_to(product_folder)).replace("/", " - ")
                    variants.append((variant_name, folder, files))

            direct_files = _image_files(product_folder)
            direct_subdirs = [path for path in product_folder.iterdir() if path.is_dir()]
            if direct_files and not direct_subdirs:
                variants.append(("Default", product_folder, direct_files))

            if variants:
                groups.append((product_folder, variants))
    return groups


def _media_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "image/jpeg"


def _route_terms(product_folder: Path, variant_name: str) -> set[str]:
    text = f"{product_folder.name} {variant_name}".lower()
    normalized = (
        text.replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
    )
    return set(normalized.split())


def _studio_backdrop_for(
    product_folder: Path,
    variant_name: str,
    forced_backdrop: str | None = None,
) -> Path:
    """Pick a local studio backdrop based on product family and variant color."""
    if forced_backdrop:
        return Path(forced_backdrop)

    product_type = _product_type_for_folder(product_folder)
    terms = _route_terms(product_folder, variant_name)
    product_name = product_folder.name.lower()

    if product_type == "perfume":
        return _backdrop_path("beige_plaster")
    if product_type == "glasses":
        return _backdrop_path("cool_gray")
    if terms & LIGHT_OR_METALLIC_TERMS:
        return _backdrop_path("charcoal")
    if "gold" in terms and not {"brown", "tan", "monogram"} & terms:
        return _backdrop_path("charcoal")
    if "novelty" in product_name:
        return _backdrop_path("cool_gray")
    if terms & PRINT_TERMS:
        return _backdrop_path("default")
    if terms & WARM_COLOR_TERMS:
        return _backdrop_path("cool_gray")
    if {"black", "navy", "blue", "green"} & terms:
        return _backdrop_path("beige_plaster")
    return _backdrop_path("default")


def _result_summary(folder: Path, result) -> dict:
    publishable = [
        review.path
        for review in result.candidate_reviews
        if review.publishable and review.silhouette_ok and review.proportions_ok
    ]
    return {
        "folder": str(folder),
        "title": result.listing.title,
        "product_type": result.prompt_route.product_type
        if result.prompt_route
        else None,
        "product_kind": result.prompt_route.product_kind if result.prompt_route else None,
        "white_bg_path": result.white_bg_path,
        "lifestyle_candidate_paths": result.lifestyle_candidate_paths,
        "on_model_candidate_paths": result.on_model_candidate_paths,
        "publishable_candidate_paths": publishable,
        "image_errors": result.image_errors,
        "shopify_product_id": result.shopify_product_id,
        "shopify_admin_url": result.shopify_admin_url,
    }


def _shopify_image_paths(result) -> list[str]:
    paths: list[str] = []
    if result.white_bg_path:
        paths.append(result.white_bg_path)
    generated = set(result.lifestyle_candidate_paths + result.on_model_candidate_paths)
    publishable = [
        review.path
        for review in result.candidate_reviews
        if (
            review.publishable
            and review.silhouette_ok
            and review.proportions_ok
            and review.path in generated
        )
    ]
    if publishable:
        paths.extend(publishable)
    elif generated:
        paths.extend(sorted(generated))
    return paths


def _basic_listing(product_folder: Path, variant_names: list[str]) -> Listing:
    product_name = product_folder.name
    variant_text = ", ".join(variant_names[:12])
    if len(variant_names) > 12:
        variant_text += f", and {len(variant_names) - 12} more"
    product_type = _product_type_for_folder(product_folder)
    category = {
        "bag": "bag",
        "perfume": "fragrance",
        "glasses": "eyewear",
    }.get(product_type, "product")
    description = (
        f"<p>{product_name} is available in multiple variants. Choose your "
        "preferred option before checkout.</p>"
        f"<ul><li>Product type: {category}</li>"
        f"<li>Available variants: {variant_text}</li>"
        "<li>Draft listing generated from organized product photos</li></ul>"
    )
    return Listing(
        title=product_name,
        description_html=description,
        seo_title=f"{product_name} | {settings.brand_name}",
        seo_description=f"Shop {product_name} in multiple variants including {variant_text}.",
        tags=[product_name.lower(), category, "variant product", *variant_names],
    )


def _product_type_for_folder(folder: Path) -> str:
    parts = set(folder.parts)
    if "perfumes" in parts:
        return "perfume"
    if "glasses" in parts:
        return "glasses"
    return "bag"


def _save_output(image_bytes: bytes, run_dir: Path, name: str) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    path.write_bytes(image_bytes)
    return str(path)


def _output_stem(index: int, source_path: Path) -> str:
    safe_stem = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in source_path.stem
    )
    return f"{index + 1:02d}_{safe_stem}"


async def _prepare_variant_images(
    product_folder: Path,
    variant_name: str,
    files: list[Path],
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[tuple[bytes, str]], list[dict], list[tuple[bytes, str]]]:
    uploads: list[tuple[bytes, str]] = []
    prepared: list[dict] = []
    generation_references: list[tuple[bytes, str]] = []

    studio_backdrop = (
        _studio_backdrop_for(product_folder, variant_name, args.backdrop)
        if args.image_source == "studio"
        else None
    )
    if studio_backdrop and not studio_backdrop.exists():
        raise RuntimeError(f"Studio backdrop not found: {studio_backdrop}")

    for image_index, path in enumerate(files):
        source_bytes = path.read_bytes()
        media_type = _media_type(path)
        stem = _output_stem(image_index, path)
        record: dict = {
            "source_path": str(path),
            "upload_path": None,
            "cutout_path": None,
            "white_bg_path": None,
            "studio_path": None,
            "backdrop_path": str(studio_backdrop) if studio_backdrop else None,
            "image_errors": [],
        }

        try:
            if args.image_source == "original":
                uploads.append((source_bytes, variant_name))
                generation_references.append((source_bytes, media_type))
                record["upload_path"] = str(path)
            else:
                cutout = await images.remove_background(source_bytes, media_type)
                record["cutout_path"] = _save_output(cutout, run_dir, f"{stem}_cutout.png")
                generation_references.append((cutout, "image/png"))

                if args.image_source == "white-bg":
                    final = images.composite_on_white(cutout)
                    record["white_bg_path"] = _save_output(
                        final, run_dir, f"{stem}_white_bg.png"
                    )
                    uploads.append((final, variant_name))
                    record["upload_path"] = record["white_bg_path"]
                else:
                    final = images.compose_on_backdrop(
                        cutout,
                        studio_backdrop,
                        size=args.studio_size,
                        fill=args.studio_fill,
                        shadow_opacity=args.shadow_opacity,
                        shadow_blur=args.shadow_blur,
                        shadow_offset=args.shadow_offset,
                        image_format=args.studio_format,
                        quality=args.studio_quality,
                    )
                    studio_name = f"{stem}_studio.{args.studio_format}"
                    record["studio_path"] = _save_output(final, run_dir, studio_name)
                    uploads.append((final, variant_name))
                    record["upload_path"] = record["studio_path"]
        except Exception as exc:
            record["image_errors"].append(f"{args.image_source}: {exc}")

        prepared.append(record)

    return uploads, prepared, generation_references


async def _process_one(folder: Path, files: list[Path], args: argparse.Namespace) -> dict:
    references = [(path.read_bytes(), _media_type(path)) for path in files]
    result = await pipeline.run(
        references[0][0],
        references[0][1],
        reference_images=references,
        creative_mode=args.creative,
        qc_enabled=args.qc,
        lifestyle_count=args.lifestyle_candidates,
        on_model_count=args.on_model_candidates,
    )
    return _result_summary(folder, result)


_STYLE_EXAMPLES_CACHE: list[dict] | None = None


def _style_examples() -> list[dict]:
    """Load the tagged golden-image manifest once per run."""
    global _STYLE_EXAMPLES_CACHE
    if _STYLE_EXAMPLES_CACHE is None:
        _STYLE_EXAMPLES_CACHE = image_examples.load_image_examples(
            settings.image_examples_path
        )
    return _STYLE_EXAMPLES_CACHE


def _style_reference_for(
    product_type: str, tier: str, enabled: bool
) -> list[tuple[bytes, str]] | None:
    """Return a single same-category golden of `tier` for generation, or None."""
    if not enabled:
        return None
    reference = image_examples.select_style_reference(
        _style_examples(), product_type, tier
    )
    return [(reference, "image/png")] if reference else None


async def _process_group(
    product_folder: Path,
    variants: list[tuple[str, Path, list[Path]]],
    args: argparse.Namespace,
) -> dict:
    variant_names = [name for name, _, _ in variants]
    variant_summaries: list[dict] = []
    image_bytes: list[tuple[bytes, str]] = []
    product_type = _product_type_for_folder(product_folder)
    prompt_pack = get_prompt_pack(product_type)
    listing = _basic_listing(product_folder, variant_names)
    listing_source = "basic"
    listing_error: str | None = None
    if getattr(args, "listing_source", "model") == "model":
        try:
            first_files = variants[0][2]
            first_image = first_files[0]
            first_image_bytes = first_image.read_bytes()
            first_media_type = _media_type(first_image)
            product_kind = {
                "bag": "handbag",
                "perfume": "fragrance",
                "glasses": "eyewear",
            }.get(product_type, product_type)
            current_product_context = ListingGenerationContext(
                product_family_name=product_folder.name,
                variant_names=variant_names[:100],
                variant_count=len(variant_names),
            )
            analysis = vision.classify_product(
                first_image_bytes,
                first_media_type,
                prompt_pack,
                product_kind,
            )
            listing = vision.write_listing(
                analysis,
                first_image_bytes,
                first_media_type,
                prompt_pack,
                additional_context=current_product_context,
            )
            listing_source = "model"
        except Exception as exc:
            listing_source = "basic_fallback"
            listing_error = str(exc)

    for index, (variant_name, folder, files) in enumerate(variants):
        # Lifestyle shows the per-color product cleanly, so generate one for EVERY
        # color. On-model conveys scale / how it's carried — the same across colors —
        # so generate it only for the hero (first) variant.
        do_lifestyle = args.creative in {"lifestyle", "both"}
        do_on_model = args.creative in {"model", "both"} and index == 0
        run_dir = Path(settings.output_dir) / uuid.uuid4().hex[:8]
        image_errors: list[str] = []
        white_bg_path = None
        lifestyle_paths: list[str] = []
        on_model_paths: list[str] = []
        processed_images: list[dict] = []

        prepared_uploads, processed_images, generation_references = (
            await _prepare_variant_images(product_folder, variant_name, files, run_dir, args)
        )
        image_bytes.extend(prepared_uploads)
        image_errors.extend(
            error
            for record in processed_images
            for error in record["image_errors"]
        )
        white_bg_paths = [
            record["white_bg_path"]
            for record in processed_images
            if record.get("white_bg_path")
        ]
        white_bg_path = white_bg_paths[0] if white_bg_paths else None

        if generation_references and do_lifestyle:
            try:
                for i, candidate in enumerate(
                    await images.generate_lifestyle(
                        generation_references,
                        prompt_pack,
                        count=args.lifestyle_candidates,
                        style_references=_style_reference_for(
                            product_type,
                            "lifestyle",
                            getattr(args, "style_reference", False),
                        ),
                    )
                ):
                    lifestyle_paths.append(
                        _save_output(candidate, run_dir, f"lifestyle_{i}.png")
                    )
                    image_bytes.append((candidate, variant_name))
            except Exception as exc:
                image_errors.append(f"lifestyle: {exc}")

        if generation_references and do_on_model:
            try:
                product_kind = "purse" if product_type == "bag" else product_type
                for i, candidate in enumerate(
                    await images.generate_on_model(
                        generation_references,
                        product_kind=product_kind,
                        prompt_pack=prompt_pack,
                        count=args.on_model_candidates,
                        style_references=_style_reference_for(
                            product_type,
                            "on_model",
                            getattr(args, "style_reference", False),
                        ),
                    )
                ):
                    on_model_paths.append(
                        _save_output(candidate, run_dir, f"on_model_{i}.png")
                    )
                    image_bytes.append((candidate, variant_name))
            except Exception as exc:
                image_errors.append(f"on_model: {exc}")

        summary = {
            "folder": str(folder),
            "variant_name": variant_name,
            "white_bg_path": white_bg_path,
            "processed_images": processed_images,
            "lifestyle_candidate_paths": lifestyle_paths,
            "on_model_candidate_paths": on_model_paths,
            "image_errors": image_errors,
        }
        if not prepared_uploads:
            summary["fatal_error"] = "No uploadable images available"
        variant_summaries.append(summary)

    if not image_bytes:
        raise RuntimeError(f"No generated images available for {product_folder}")

    created = await shopify.create_draft_with_variants(
        listing,
        variant_names,
        image_bytes,
    )
    return {
        "folder": str(product_folder),
        "variant_names": variant_names,
        "variant_count": len(variant_names),
        "listing_source": listing_source,
        "listing_error": listing_error,
        "variant_results": variant_summaries,
        "shopify_product_id": created["product_id"],
        "shopify_admin_url": created["admin_url"],
        "image_errors": [f"shopify: {error}" for error in created["image_errors"]],
    }


async def _run(args: argparse.Namespace) -> int:
    roots = [Path(root) for root in (args.root or DEFAULT_ROOTS)]
    products = (
        _ready_product_folders(roots)
        if args.mode == "leaf"
        else _variant_groups(roots)
    )
    if args.start_after:
        products = [
            item for item in products if str(item[0]) > args.start_after
        ]

    if args.mode == "variants" and args.variant_limit:
        products = [
            (folder, variants[: args.variant_limit])
            for folder, variants in products
        ]

    if not args.all:
        products = products[: args.limit]

    print(f"ready product folders selected: {len(products)}", flush=True)
    if args.dry_run:
        for folder, payload in products:
            if args.mode == "leaf":
                print(f"{len(payload)} refs  {folder}")
            else:
                print(f"{len(payload)} variants  {folder}")
                for name, variant_folder, files in payload:
                    suffix = ""
                    if args.image_source == "studio":
                        suffix = (
                            "  backdrop="
                            f"{_studio_backdrop_for(folder, name, args.backdrop)}"
                        )
                    print(f"  - {name}: {len(files)} refs  {variant_folder}{suffix}")
        return 0

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with Path(args.manifest).open("a", encoding="utf-8") as manifest:
        for index, (folder, files) in enumerate(products, start=1):
            print(
                f"RUN {index}/{len(products)}: {folder}",
                flush=True,
            )
            try:
                summary = (
                    await _process_one(folder, files, args)
                    if args.mode == "leaf"
                    else await _process_group(folder, files, args)
                )
            except Exception as exc:
                failures += 1
                summary = {"folder": str(folder), "fatal_error": str(exc)}
            manifest.write(json.dumps(summary, ensure_ascii=True) + "\n")
            manifest.flush()
            print(json.dumps(summary, indent=2), flush=True)
    return 1 if failures else 0


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=["variants", "leaf"], default="variants")
    parser.add_argument("--variant-limit", type=int)
    parser.add_argument(
        "--listing-source",
        choices=["model", "basic"],
        default="model",
        help="Use Claude with approved examples, or the basic no-model template.",
    )
    parser.add_argument(
        "--creative",
        choices=["none", "lifestyle", "model", "both"],
        default="none",
    )
    parser.add_argument("--creative-variant-limit", type=int, default=1)
    parser.add_argument("--lifestyle-candidates", type=int, default=1)
    parser.add_argument("--on-model-candidates", type=int, default=1)
    parser.add_argument(
        "--style-reference",
        dest="style_reference",
        action="store_true",
        help=(
            "UNSAFE on gpt-image-2: feeds an approved store image as a visual "
            "reference, but the exemplar's product substitutes for the real one "
            "(product bleed). Off by default; use text style-spec instead."
        ),
    )
    parser.add_argument("--qc", action="store_true")
    parser.add_argument(
        "--image-source",
        choices=["white-bg", "studio", "original"],
        default="white-bg",
    )
    parser.add_argument("--backdrop")
    parser.add_argument("--studio-size", type=int, default=2000)
    parser.add_argument("--studio-fill", type=float, default=0.72)
    parser.add_argument("--shadow-opacity", type=int, default=90)
    parser.add_argument("--shadow-blur", type=int, default=40)
    parser.add_argument("--shadow-offset", type=float, default=0.025)
    parser.add_argument("--studio-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--studio-quality", type=int, default=90)
    parser.add_argument("--start-after", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=f"outputs/batch_{stamp}.jsonl")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
