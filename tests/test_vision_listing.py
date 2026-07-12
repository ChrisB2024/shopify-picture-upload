"""Listing-generation output safety contracts."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.product_types.prompts import BAG_PROMPT_PACK
from app.schemas import Listing, ProductAnalysis
from app.services import vision


class VisionListingSafetyTests(unittest.TestCase):
    def test_write_listing_sanitizes_model_generated_description_html(self) -> None:
        analysis = ProductAnalysis(
            product_type="bag",
            product_kind="handbag",
            primary_color="black",
            material="leather",
            style_keywords=["structured", "polished", "modern"],
            notable_features=["top handle"],
        )
        model_listing = Listing(
            title="Safe Output Bag",
            description_html=(
                '<div onclick="outer()"><h2 onclick="heading()">Heading</h2>'
                '<p class="copy" onclick="paragraph()">Safe '
                '<strong style="color:red" onclick="bold()">bold</strong>'
                '<script>UNSAFE_SCRIPT()</script>'
                '<a href="https://bad.test" onclick="link()">link</a>'
                '<br onclick="break()">tail</p>'
                '<ul onclick="list()"><li onmouseover="item()">Item</li></ul>'
                '<img src="bad.jpg" onerror="image()"></div>'
            ),
            seo_title="Safe Output Bag | OHH Bags",
            seo_description="Explore the Safe Output Bag.",
            tags=["bag", "black", "structured"],
        )

        with (
            patch.object(vision, "load_listing_examples", return_value=[]),
            patch.object(vision, "_create_parsed", return_value=model_listing),
        ):
            listing = vision.write_listing(
                analysis,
                b"reference-image",
                "image/jpeg",
                BAG_PROMPT_PACK,
            )

        self.assertEqual(
            listing.description_html,
            "<h2>Heading</h2><p>Safe <strong>bold</strong>link<br>tail</p>"
            "<ul><li>Item</li></ul>",
        )
        self.assertNotIn("script", listing.description_html.casefold())
        self.assertNotIn("onclick", listing.description_html.casefold())
        self.assertNotIn("onmouseover", listing.description_html.casefold())
        self.assertNotIn("onerror", listing.description_html.casefold())
        self.assertNotIn("<a", listing.description_html.casefold())
        self.assertNotIn("<img", listing.description_html.casefold())
        self.assertEqual(listing.title, model_listing.title)
        self.assertEqual(listing.tags, model_listing.tags)


if __name__ == "__main__":
    unittest.main()
