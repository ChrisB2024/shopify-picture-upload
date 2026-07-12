"""Grouped-batch contracts for model-backed versus basic listing copy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app import batch
from app.schemas import Listing, ListingGenerationContext, ProductAnalysis


def batch_args(listing_source: str) -> SimpleNamespace:
    """Return only the batch options exercised by `_process_group`."""
    return SimpleNamespace(
        listing_source=listing_source,
        creative="none",
        creative_variant_limit=0,
        lifestyle_candidates=1,
        on_model_candidates=1,
    )


class GroupedListingSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_source_writes_and_uploads_the_returned_listing(self) -> None:
        analysis = ProductAnalysis(
            product_type="bag",
            product_kind="handbag",
            primary_color="red",
            material="leather",
            style_keywords=["structured", "polished", "modern"],
            notable_features=["top handle"],
        )
        model_listing = Listing(
            title="The Model-Written Voyager",
            description_html="<p>Grounded model-written product copy.</p>",
            seo_title="The Model-Written Voyager | OHH Bags",
            seo_description="Explore the model-written Voyager bag.",
            tags=["bag", "voyager", "structured"],
        )
        upload = (b"prepared-image", "Red")
        prepared = (
            [upload],
            [{"white_bg_path": "prepared.png", "image_errors": []}],
            [(b"reference", "image/png")],
        )
        created = {
            "product_id": "shopify-123",
            "admin_url": "https://example.test/admin/products/shopify-123",
            "image_errors": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            product_folder = Path(temp_dir) / "incoming" / "bags" / "Voyager"
            variant_folder = product_folder / "Red"
            variant_folder.mkdir(parents=True)
            first_image = variant_folder / "front.jpg"
            first_image.write_bytes(b"first-reference-image")
            variants = [("Red", variant_folder, [first_image])]

            classify = Mock(return_value=analysis)
            write_listing = Mock(return_value=model_listing)
            create_draft = AsyncMock(return_value=created)
            with (
                patch.object(batch.vision, "classify_product", classify),
                patch.object(batch.vision, "write_listing", write_listing),
                patch.object(
                    batch,
                    "_prepare_variant_images",
                    AsyncMock(return_value=prepared),
                ),
                patch.object(
                    batch.shopify,
                    "create_draft_with_variants",
                    create_draft,
                ),
            ):
                result = await batch._process_group(
                    product_folder,
                    variants,
                    batch_args("model"),
                )

        classify.assert_called_once()
        write_listing.assert_called_once()
        write_args, write_kwargs = write_listing.call_args
        self.assertIs(write_args[0], analysis)
        self.assertEqual(write_args[1], b"first-reference-image")
        self.assertEqual(write_args[2], "image/jpeg")
        self.assertIn("additional_context", write_kwargs)
        context = write_kwargs["additional_context"]
        self.assertIsInstance(context, ListingGenerationContext)
        self.assertEqual(context.product_family_name, "Voyager")
        self.assertEqual(context.variant_names, ["Red"])
        self.assertEqual(context.variant_count, 1)
        create_draft.assert_awaited_once_with(model_listing, ["Red"], [upload])
        self.assertEqual(result["listing_source"], "model")
        self.assertIsNone(result["listing_error"])
        self.assertEqual(result["shopify_product_id"], "shopify-123")

    async def test_basic_source_skips_model_and_uploads_basic_listing(self) -> None:
        upload = (b"prepared-image", "Black")
        prepared = (
            [upload],
            [{"white_bg_path": "prepared.png", "image_errors": []}],
            [(b"reference", "image/png")],
        )
        created = {
            "product_id": "shopify-456",
            "admin_url": "https://example.test/admin/products/shopify-456",
            "image_errors": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            product_folder = Path(temp_dir) / "incoming" / "bags" / "Basic Bag"
            variant_folder = product_folder / "Black"
            variant_folder.mkdir(parents=True)
            first_image = variant_folder / "front.jpg"
            first_image.write_bytes(b"unused-by-basic-listing")
            variants = [("Black", variant_folder, [first_image])]

            classify = Mock()
            write_listing = Mock()
            create_draft = AsyncMock(return_value=created)
            with (
                patch.object(batch.vision, "classify_product", classify),
                patch.object(batch.vision, "write_listing", write_listing),
                patch.object(
                    batch,
                    "_prepare_variant_images",
                    AsyncMock(return_value=prepared),
                ),
                patch.object(
                    batch.shopify,
                    "create_draft_with_variants",
                    create_draft,
                ),
            ):
                result = await batch._process_group(
                    product_folder,
                    variants,
                    batch_args("basic"),
                )

        classify.assert_not_called()
        write_listing.assert_not_called()
        create_draft.assert_awaited_once()
        uploaded_listing, variant_names, uploads = create_draft.await_args.args
        self.assertIsInstance(uploaded_listing, Listing)
        self.assertEqual(uploaded_listing.title, "Basic Bag")
        self.assertEqual(variant_names, ["Black"])
        self.assertEqual(uploads, [upload])
        self.assertEqual(result["listing_source"], "basic")
        self.assertIsNone(result["listing_error"])


if __name__ == "__main__":
    unittest.main()
