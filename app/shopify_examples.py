"""Export approved Shopify listings as local few-shot examples.

The export is deliberately read-only. It keeps a privacy-minimized source
snapshot for audit/debugging and a model-ready JSONL file aligned with
``Listing``.

Usage:
    python -m app.shopify_examples \
      --start-title "Luxe Voyager Set" \
      --end-title "The Rebel Muse Bag"
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.schemas import Listing
from app.services import shopify
from app.services.listing_examples import (
    product_to_listing_example,
    select_inclusive_range,
)

DEFAULT_OUTPUT_DIR = Path("outputs/shopify_examples")
DEFAULT_START_TITLE = "Luxe Voyager Set"
DEFAULT_END_TITLE = "The Rebel Muse Bag"
DEFAULT_PRODUCT_TYPE = "bag"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    """Retry bounded Shopify read failures and honor numeric Retry-After."""
    last_exception: Exception | None = None
    for attempt in range(3):
        try:
            response = await client.request(method, url, **kwargs)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exception = exc
            if attempt == 2:
                raise RuntimeError(f"Shopify read connection failed: {exc}") from exc
            await asyncio.sleep(2**attempt)
            continue

        if response.status_code not in _RETRYABLE_STATUS_CODES or attempt == 2:
            return response
        retry_after = response.headers.get("retry-after", "")
        try:
            delay = max(0.0, min(float(retry_after), 10.0))
        except ValueError:
            delay = float(2**attempt)
        await asyncio.sleep(delay)

    raise RuntimeError(f"Shopify read failed: {last_exception}")


async def _fetch_products(status: str = "active") -> list[dict[str, Any]]:
    """Fetch every Shopify product page for ``status`` without mutating Shopify."""
    token = await shopify.get_access_token()
    headers = {"X-Shopify-Access-Token": token}
    url = (
        f"https://{settings.shopify_store}/admin/api/"
        f"{settings.shopify_api_version}/products.json"
    )
    params: dict[str, str | int] | None = {"status": status, "limit": 250}
    products: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=90) as client:
        while url:
            response = await _request_with_retry(
                client,
                "GET",
                url,
                headers=headers,
                params=params,
            )
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(
                    f"Shopify product fetch failed {response.status_code}: "
                    f"{response.text}"
                )
            products.extend(response.json().get("products", []))
            next_link = response.links.get("next", {})
            url = str(next_link.get("url") or "")
            params = None

    return products


async def _fetch_product_seo(
    product_ids: list[str | int],
) -> dict[str, dict[str, str | None]]:
    """Fetch Shopify's explicit SEO fields for product IDs via Admin GraphQL."""
    if not product_ids:
        return {}

    token = await shopify.get_access_token()
    url = (
        f"https://{settings.shopify_store}/admin/api/"
        f"{settings.shopify_api_version}/graphql.json"
    )
    headers = {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
    }
    query = """
      query ProductSeo($ids: [ID!]!) {
        nodes(ids: $ids) {
          ... on Product {
            id
            seo { title description }
          }
        }
      }
    """
    result: dict[str, dict[str, str | None]] = {}

    async with httpx.AsyncClient(timeout=90) as client:
        for offset in range(0, len(product_ids), 100):
            chunk = product_ids[offset : offset + 100]
            gids = [f"gid://shopify/Product/{product_id}" for product_id in chunk]
            response = await _request_with_retry(
                client,
                "POST",
                url,
                headers=headers,
                json={"query": query, "variables": {"ids": gids}},
            )
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(
                    f"Shopify SEO fetch failed {response.status_code}: {response.text}"
                )
            payload = response.json()
            if payload.get("errors"):
                raise RuntimeError(
                    "Shopify SEO fetch returned GraphQL errors: "
                    f"{json.dumps(payload['errors'], ensure_ascii=False)}"
                )
            for node in payload.get("data", {}).get("nodes", []):
                if not node:
                    continue
                legacy_id = str(node["id"]).rsplit("/", 1)[-1]
                seo = node.get("seo") or {}
                result[legacy_id] = {
                    "title": seo.get("title"),
                    "description": seo.get("description"),
                }

    return result


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def _raw_product_snapshot(
    product: dict[str, Any],
    *,
    seo: dict[str, str | None],
    range_index: int,
) -> dict[str, Any]:
    """Keep useful catalog provenance while excluding inventory/admin internals."""
    return {
        "id": str(product.get("id") or ""),
        "admin_graphql_api_id": product.get("admin_graphql_api_id"),
        "range_index": range_index,
        "title": product.get("title"),
        "handle": product.get("handle"),
        "status": product.get("status"),
        "body_html": product.get("body_html"),
        "vendor": product.get("vendor"),
        "product_type": product.get("product_type"),
        "tags": product.get("tags"),
        "created_at": product.get("created_at"),
        "updated_at": product.get("updated_at"),
        "published_at": product.get("published_at"),
        "shopify_seo": seo,
        "options": [
            {
                key: option.get(key)
                for key in ("id", "name", "position", "values")
                if key in option
            }
            for option in product.get("options", [])
        ],
        "variants": [
            {
                key: variant.get(key)
                for key in ("id", "title", "option1", "option2", "option3")
                if key in variant
            }
            for variant in product.get("variants", [])
        ],
        "images": [
            {
                key: image.get(key)
                for key in (
                    "id",
                    "alt",
                    "position",
                    "width",
                    "height",
                    "src",
                    "variant_ids",
                )
                if key in image
            }
            for image in product.get("images", [])
        ],
    }


async def export_examples(
    *,
    start_title: str,
    end_title: str,
    output_dir: Path,
    status: str = "active",
    product_type_hint: str = DEFAULT_PRODUCT_TYPE,
) -> dict[str, Any]:
    """Fetch, validate, and write one inclusive chronological product range."""
    if status != "active":
        raise ValueError("Gold listing examples must be exported with status='active'")

    products = await _fetch_products(status=status)
    selected = select_inclusive_range(
        products,
        start_title,
        end_title,
        order_key="created_at",
    )
    seo_by_id = await _fetch_product_seo([product["id"] for product in selected])

    raw_records: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for index, product in enumerate(selected, start=1):
        product_id = str(product["id"])
        seo = seo_by_id.get(product_id, {})
        raw_record = _raw_product_snapshot(
            product,
            seo=seo,
            range_index=index,
        )
        raw_records.append(raw_record)

        example = product_to_listing_example(
            product,
            seo=seo,
            range_index=index,
            product_type_hint=product_type_hint,
        )
        # Validate the exact payload later supplied to Claude.
        Listing.model_validate(example["listing"])
        examples.append(example)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    raw_path = output_dir / "products.raw.jsonl"
    examples_path = output_dir / "listing_examples.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_jsonl(raw_path, raw_records)
    _write_jsonl(examples_path, examples)

    exported_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "exported_at": exported_at,
        "shopify_store": settings.shopify_store,
        "shopify_api_version": settings.shopify_api_version,
        "status": status,
        "product_type_hint": product_type_hint,
        "order": ["created_at", "id"],
        "inclusive": True,
        "requested_start_title": start_title,
        "requested_end_title": end_title,
        "matched_start_title": selected[0]["title"],
        "matched_end_title": selected[-1]["title"],
        "product_count": len(selected),
        "eligible_example_count": sum(
            bool(example.get("quality", {}).get("eligible")) for example in examples
        ),
        "field_coverage": {
            "description_html": sum(
                bool(example["listing"].get("description_html"))
                for example in examples
            ),
            "explicit_shopify_seo_title": sum(
                example["quality"].get("seo_title_source") == "shopify"
                for example in examples
            ),
            "explicit_shopify_seo_description": sum(
                example["quality"].get("seo_description_source") == "shopify"
                for example in examples
            ),
            "nonempty_tags": sum(
                bool(example["listing"].get("tags")) for example in examples
            ),
            "product_type": sum(
                bool(example.get("product_type")) for example in examples
            ),
        },
        "products": [
            {
                "range_index": index,
                "id": str(product["id"]),
                "title": product["title"],
                "created_at": product.get("created_at"),
            }
            for index, product in enumerate(selected, start=1)
        ],
        "files": {
            "raw": raw_path.name,
            "model_ready": examples_path.name,
        },
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.chmod(0o600)
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an inclusive active Shopify range as listing examples."
    )
    parser.add_argument("--start-title", default=DEFAULT_START_TITLE)
    parser.add_argument("--end-title", default=DEFAULT_END_TITLE)
    parser.add_argument("--product-type", default=DEFAULT_PRODUCT_TYPE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = asyncio.run(
        export_examples(
            start_title=args.start_title,
            end_title=args.end_title,
            output_dir=args.output_dir,
            product_type_hint=args.product_type,
        )
    )
    print(
        f"Exported {manifest['product_count']} active products to "
        f"{args.output_dir / 'listing_examples.jsonl'}"
    )
    print(
        f"Inclusive range: {manifest['matched_start_title']} -> "
        f"{manifest['matched_end_title']}"
    )


if __name__ == "__main__":
    main()
