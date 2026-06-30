"""Shopify Admin API: create the product as a DRAFT (never auto-publish).

Auth: Dev Dashboard apps use the client-credentials grant. We exchange
client_id + client_secret for a short-lived (~24h) Admin API token and cache it.

Draft-only rule: keep status="draft" hard-coded until you trust the output.
"""

import asyncio
import time
import base64

import httpx

from app.config import settings
from app.schemas import Listing

# Simple in-process token cache: {"token": str, "expires_at": epoch_seconds}.
_token: dict = {}

_TRANSIENT_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)


async def get_access_token() -> str:
    """Return a valid Admin API token, exchanging client creds if needed.

    Client-credentials grant:
        POST https://{store}/admin/oauth/access_token
        body (form-encoded): grant_type=client_credentials,
                             client_id=..., client_secret=...
        -> {"access_token": ..., "scope": ..., "expires_in": 86399}
    Only works when the app and store are in the same Shopify org.
    """
    now = time.time()
    if _token.get("token") and _token.get("expires_at", 0) > now + 60:
        return _token["token"]

    if not (settings.shopify_client_id and settings.shopify_client_secret):
        raise RuntimeError("SHOPIFY_CLIENT_ID / SHOPIFY_CLIENT_SECRET not set")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"https://{settings.shopify_store}/admin/oauth/access_token",
            data={
                "grant_type": "client_credentials",
                "client_id": settings.shopify_client_id,
                "client_secret": settings.shopify_client_secret,
            },
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Shopify token exchange failed {resp.status_code}: {resp.text}")

    data = resp.json()
    _token["token"] = data["access_token"]
    _token["expires_at"] = now + data.get("expires_in", 86399)
    return _token["token"]


async def create_draft(
    listing: Listing,
    image_bytes: bytes | list[bytes] | None,
) -> dict:
    """Create a DRAFT product. Return {'product_id', 'admin_url'}.

    Auth is handled — call `token = await get_access_token()` and send it as the
    X-Shopify-Access-Token header.

    Product creation:
      POST https://{store}/admin/api/{version}/products.json
      headers: {"X-Shopify-Access-Token": token, "Content-Type": "application/json"}
      body:
        {"product": {
            "title": listing.title,
            "body_html": listing.description_html,
            "tags": listing.tags,                 # list or comma-joined string
            "status": "draft",                    # <-- non-negotiable for now
            "metafields_global_title_tag": listing.seo_title,
            "metafields_global_description_tag": listing.seo_description,
        }}

    Image upload happens after product creation, one image per request:
      POST https://{store}/admin/api/{version}/products/{product_id}/images.json

    """
    if not settings.shopify_store:
        raise RuntimeError("SHOPIFY_STORE not set")

    token = await get_access_token()
    product: dict = {
        "title": listing.title,
        "body_html": listing.description_html,
        "tags": ", ".join(listing.tags),
        "status": "draft",
        "metafields_global_title_tag": listing.seo_title,
        "metafields_global_description_tag": listing.seo_description,
    }

    url = (
        f"https://{settings.shopify_store}/admin/api/"
        f"{settings.shopify_api_version}/products.json"
    )
    headers = {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
    }
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=90) as client:
        for attempt in range(2):
            try:
                resp = await client.post(
                    url,
                    headers=headers,
                    json={"product": product},
                )
                break
            except _TRANSIENT_ERRORS as exc:
                last_exc = exc
                if attempt == 1:
                    raise RuntimeError(
                        f"Shopify product create connection failed: {exc}"
                    ) from exc
                await asyncio.sleep(1)
        else:
            raise RuntimeError(f"Shopify product create failed: {last_exc}")

    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"Shopify product create failed {resp.status_code}: {resp.text}")

    data = resp.json()
    product_id = str(data["product"]["id"])
    image_errors = await upload_product_images(product_id, image_bytes, token=token)
    return {
        "product_id": product_id,
        "admin_url": f"https://{settings.shopify_store}/admin/products/{product_id}",
        "image_errors": image_errors,
    }


async def create_draft_with_variants(
    listing: Listing,
    variant_names: list[str],
    image_bytes: bytes | list[bytes] | None,
    *,
    option_name: str = "Color or Print",
) -> dict:
    """Create one DRAFT product with Shopify variants.

    The generated images are attached at product level. Variant-specific image
    assignment can be added later after upload IDs are available, but this
    prevents separate color/print folders from becoming separate products.
    """
    if not settings.shopify_store:
        raise RuntimeError("SHOPIFY_STORE not set")
    if not variant_names:
        raise RuntimeError("At least one variant is required")

    token = await get_access_token()
    product: dict = {
        "title": listing.title,
        "body_html": listing.description_html,
        "tags": ", ".join(listing.tags),
        "status": "draft",
        "metafields_global_title_tag": listing.seo_title,
        "metafields_global_description_tag": listing.seo_description,
        "options": [{"name": option_name, "values": variant_names}],
        "variants": [{"option1": name} for name in variant_names],
    }

    url = (
        f"https://{settings.shopify_store}/admin/api/"
        f"{settings.shopify_api_version}/products.json"
    )
    headers = {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
    }
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=90) as client:
        for attempt in range(2):
            try:
                resp = await client.post(
                    url,
                    headers=headers,
                    json={"product": product},
                )
                break
            except _TRANSIENT_ERRORS as exc:
                last_exc = exc
                if attempt == 1:
                    raise RuntimeError(
                        f"Shopify variant product create connection failed: {exc}"
                    ) from exc
                await asyncio.sleep(1)
        else:
            raise RuntimeError(f"Shopify variant product create failed: {last_exc}")

    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"Shopify variant product create failed {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    product_id = str(data["product"]["id"])
    image_errors = await upload_product_images(product_id, image_bytes, token=token)
    return {
        "product_id": product_id,
        "admin_url": f"https://{settings.shopify_store}/admin/products/{product_id}",
        "image_errors": image_errors,
    }


async def upload_product_images(
    product_id: str,
    image_bytes: bytes | list[bytes] | None,
    *,
    token: str | None = None,
) -> list[str]:
    """Attach images to an existing product one request at a time."""
    if not image_bytes:
        return []

    token = token or await get_access_token()
    image_list = image_bytes if isinstance(image_bytes, list) else [image_bytes]
    url = (
        f"https://{settings.shopify_store}/admin/api/"
        f"{settings.shopify_api_version}/products/{product_id}/images.json"
    )
    headers = {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
    }
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=90) as client:
        for index, image in enumerate(image_list):
            payload = {
                "image": {
                    "attachment": base64.b64encode(image).decode("ascii"),
                    "position": index + 1,
                }
            }
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    break
                except _TRANSIENT_ERRORS as exc:
                    last_exc = exc
                    if attempt == 1:
                        errors.append(f"image {index + 1}: connection failed: {exc}")
                    else:
                        await asyncio.sleep(1)
            else:
                errors.append(f"image {index + 1}: failed: {last_exc}")
                continue

            if resp.status_code < 200 or resp.status_code >= 300:
                errors.append(
                    f"image {index + 1}: Shopify upload failed "
                    f"{resp.status_code}: {resp.text}"
                )

    return errors
