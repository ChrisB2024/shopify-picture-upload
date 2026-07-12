"""Golden product-image corpus: approved store images used as style references.

Mirrors the copy-example flow (shopify_examples.py + listing_examples.py) but for
images. Two responsibilities:

  - build_corpus(): download + tag the approved store images (run once, offline).
  - load_image_examples() / select_style_reference(): pick a same-category golden
    of a given tier at generation time.

The build reads products.raw.jsonl (produced by `python -m app.shopify_examples`),
which already contains each product's image `src` URLs and product_type — so we
don't re-hit the Shopify Admin API here, just the public image CDN.

Run the build:
    python -m app.services.image_examples
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from app.config import settings
from app.services.listing_examples import _normalized_product_type

_openai = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

VALID_TIERS = {"white_bg", "lifestyle", "on_model", "other"}
DEFAULT_RAW = Path("outputs/shopify_examples/products.raw.jsonl")
DEFAULT_IMAGES_DIR = Path("outputs/shopify_examples/images")
DEFAULT_MANIFEST = Path("outputs/shopify_examples/image_examples.jsonl")

_TIER_SYSTEM_PROMPT = (
    "Classify an ecommerce product image into exactly one tier. "
    'Return JSON {"tier": "..."} where tier is one of:\n'
    "- white_bg: the product alone on a plain white or neutral studio background, "
    "no person and no real-world scene.\n"
    "- lifestyle: the product staged in a real scene or setting (surface, backdrop, "
    "props) but with NO person.\n"
    "- on_model: a person is wearing, holding, or carrying the product.\n"
    "- other: anything else (packaging, size chart, graphic, collage, text).\n"
    "Return only the JSON."
)


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def _product_type_map(listing_examples_path: str | Path) -> dict[str, str]:
    """Map source_product_id -> resolved product_type from the copy corpus.

    Shopify's raw `product_type` field is often blank; the copy examples already
    resolve it (with the export's hint), so we reuse that instead of the raw
    field to keep image and copy categories consistent.
    """
    mapping: dict[str, str] = {}
    for record in _read_jsonl(Path(listing_examples_path)):
        product_id = str(record.get("source_product_id") or "")
        product_type = _normalized_product_type(record.get("product_type"))
        if product_id and product_type and product_type != "other":
            mapping[product_id] = product_type
    return mapping


# --------------------------------------------------------------------------- #
# Tier classification (vision)
# --------------------------------------------------------------------------- #

def _classify_tier(image_bytes: bytes, media_type: str) -> str:
    """Return the tier for one image via a cheap OpenAI vision call."""
    if not _openai:
        return "other"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = _openai.chat.completions.create(
            model=settings.openai_vision_model,
            temperature=0,
            max_tokens=30,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _TIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded}"
                            },
                        },
                    ],
                },
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        tier = str(data.get("tier", "other")).strip()
        return tier if tier in VALID_TIERS else "other"
    except Exception:
        # Tagging is best-effort — a failure just means this image is unusable
        # as a style reference, not a pipeline error.
        return "other"


# --------------------------------------------------------------------------- #
# Corpus build (run once)
# --------------------------------------------------------------------------- #

async def _download(client: httpx.AsyncClient, src: str) -> tuple[bytes, str] | None:
    try:
        response = await client.get(src, timeout=60, follow_redirects=True)
    except httpx.HTTPError:
        return None
    if response.status_code >= 400:
        return None
    media = (response.headers.get("content-type", "") or "").split(";")[0].strip()
    return response.content, media or "image/jpeg"


async def build_corpus(
    *,
    raw_path: Path = DEFAULT_RAW,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    listing_examples_path: str | Path | None = None,
    product_type_hint: str = "bag",
    per_product: int = 8,
    limit: int | None = None,
) -> dict[str, Any]:
    """Download + tag store images into a reusable style-reference manifest.

    Idempotent: images already present in the manifest (with a file on disk) are
    skipped for the expensive download+tag, but their resolved product_type is
    refreshed cheaply, so a re-run repairs categories without re-paying.
    """
    by_image: dict[str, dict[str, Any]] = {
        str(record.get("image_id")): record
        for record in _read_jsonl(manifest_path)
    }

    type_map = _product_type_map(
        listing_examples_path or settings.listing_examples_path
    )
    hint = _normalized_product_type(product_type_hint) or "other"

    products = _read_jsonl(raw_path)
    if limit is not None:
        products = products[:limit]
    images_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        for product in products:
            product_id = str(product.get("id") or "")
            resolved = type_map.get(product_id)
            product_type = resolved if resolved and resolved != "other" else hint
            title = product.get("title") or ""
            for image in (product.get("images") or [])[:per_product]:
                image_id = str(image.get("id") or "")
                src = image.get("src")
                if not image_id or not src:
                    continue
                cached = by_image.get(image_id)
                if cached and Path(cached.get("path", "")).exists():
                    cached["product_type"] = product_type  # refresh cheap metadata
                    continue  # already downloaded + tagged

                downloaded = await _download(client, src)
                if not downloaded:
                    continue
                data, media = downloaded
                ext = mimetypes.guess_extension(media) or ".jpg"
                path = images_dir / f"{product_id}_{image_id}{ext}"
                path.write_bytes(data)

                tier = await asyncio.to_thread(_classify_tier, data, media)
                by_image[image_id] = {
                    "source_product_id": product_id,
                    "source_title": title,
                    "product_type": product_type,
                    "image_id": image_id,
                    "position": int(image.get("position") or 0),
                    "tier": tier,
                    "src": src,
                    "path": str(path),
                    "media_type": media,
                }

    records = sorted(
        by_image.values(),
        key=lambda r: (r["product_type"], r["tier"], r["source_product_id"], r["position"]),
    )
    _write_jsonl(manifest_path, records)
    return {"count": len(records), "manifest": str(manifest_path)}


# --------------------------------------------------------------------------- #
# Runtime: load + select
# --------------------------------------------------------------------------- #

def load_image_examples(path: str | Path) -> list[dict[str, Any]]:
    """Load the tagged style-reference manifest (empty list if it doesn't exist)."""
    return _read_jsonl(Path(path))


def select_style_reference(
    examples: list[dict[str, Any]],
    product_type: str,
    tier: str,
    *,
    kind: str | None = None,
) -> bytes | None:
    """Return the bytes of a same-category golden image of `tier`, or None.

    Deterministic: same-`product_type`, matching `tier`, lowest position wins.
    `kind` is accepted for future refinement (e.g. prefer purse exemplars); it
    is currently unused so a bag always matches a bag golden.
    """
    wanted = _normalized_product_type(product_type)
    candidates = [
        record
        for record in examples
        if record.get("tier") == tier
        and _normalized_product_type(record.get("product_type")) == wanted
    ]
    candidates.sort(
        key=lambda r: (int(r.get("position") or 0), str(r.get("source_product_id")))
    )
    for record in candidates:
        path = Path(record.get("path", ""))
        if path.exists():
            try:
                return path.read_bytes()
            except OSError:
                continue
    return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the golden product-image corpus used as style references."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--product-type", default="bag", help="Fallback product type.")
    parser.add_argument("--per-product", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.raw.exists():
        raise SystemExit(
            f"{args.raw} not found — run `python -m app.shopify_examples` first."
        )

    result = asyncio.run(
        build_corpus(
            raw_path=args.raw,
            images_dir=args.images_dir,
            manifest_path=args.manifest,
            product_type_hint=args.product_type,
            per_product=args.per_product,
            limit=args.limit,
        )
    )
    counts = Counter(record["tier"] for record in load_image_examples(args.manifest))
    print(f"Tagged {result['count']} images -> {result['manifest']}")
    print("Tiers:", dict(counts))


if __name__ == "__main__":
    main()
