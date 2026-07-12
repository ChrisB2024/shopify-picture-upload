# Listing example acceptance plan

Run the offline contract suite with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The suite uses synthetic Shopify records and makes no network calls. It covers:

1. Boundary titles are matched exactly after Unicode normalization, case folding,
   trimming, and whitespace collapsing; near matches are rejected.
2. Only active products are retained. The inclusive slice is sorted by the
   actual `created_at` instant and then numeric product ID, regardless of input
   order or which endpoint was supplied first.
3. A raw Shopify product plus SEO data is converted to a JSON-serializable record
   whose nested `listing` payload validates against `app.schemas.Listing`; an
   explicit product-type hint safely fills a blank Shopify product type.
4. JSONL loading validates quality metadata, rejects malformed records, and
   canonicalizes the nested listing so unrecognized keys cannot reach a prompt;
   range indexes must be finite, non-negative integers.
5. Example selection is deterministic, unique, capped at three, and favors copy
   relevant to the current product analysis. Cross-product-type and zero-overlap
   examples are excluded.
6. The few-shot prompt includes the selected examples and explicitly treats them
   as style/structure references while prohibiting reuse of their product facts.
   Data containing literal prompt delimiters is JSON-escaped.
7. Script, style, template, SVG, and hidden HTML content is excluded from both
   fallback visible text and the few-shot prompt, including malformed closing-tag
   cases. Model-generated listing HTML is reduced to attribute-free allowlisted tags.
8. Grouped batch mode sends model-written copy to Shopify when requested, while
   explicit basic mode skips both model calls and uploads the local template. The
   model path receives a typed context with exact family, variant, and count fields.

Live Shopify pagination/authentication should additionally be smoke-tested with
the configured store, because that behavior cannot be verified safely in an
offline unit test.
