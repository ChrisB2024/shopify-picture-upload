"""Product-type prompt packs.

Keep product-specific judgment here instead of growing one universal prompt.
Shared services consume these packs for classification, copy, image generation,
and QC.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductPromptPack:
    product_type: str
    classify_prompt: str
    listing_prompt: str
    fidelity_rules: str
    lifestyle_prompt: str
    model_prompt: str | None
    purse_model_prompt: str | None = None
    review_focus: str = ""


BAG_FIDELITY_RULES = """Product fidelity rules:
- If multiple reference photos are provided, they are different angles of the same exact product. Use all of them to understand the product's true 3D shape, side profile, handle/strap placement, depth, hardware, texture, and proportions. Do not combine them into multiple products.
- Preserve the exact outer silhouette and proportions of the reference product. If the reference is heart-shaped, crescent, boxy, rectangular, rounded, slouchy, structured, mini, oversized, or asymmetrical, the generated product must keep that same shape.
- Do not stretch, slim, widen, round off, square off, or simplify the bag. Do not change handle height, flap shape, clasp placement, strap attachment points, chain length, side profile, corner radius, or base width.
- Keep distinctive details in the same locations: clasp, lock, logo, seams, stitches, studs, beads, rhinestones, quilting, pleats, panels, piping, handles, strap, feet, zipper, and hardware.
- Treat the uploaded image as a product reference, not design inspiration. The scene may change, but the product design may not."""

BAG_LIFESTYLE_PROMPT = """Product picture with no model
Using the uploaded product photo as reference, create a clean, high-end ecommerce product photograph of the same handbag.

IMPORTANT:
- The handbag must match the exact design, color, texture, hardware, stitching, and proportions of the uploaded reference photo. Do not alter the product - replicate it faithfully. This is a product listing image, not a stylized reinterpretation.
- Adapt the background tone, lighting warmth, and shadow style to complement the bag's natural color palette.
- Do not add brand names, store names, text, watermarks, labels, or logos anywhere in the image.

Scene: The handbag presented on its own as the sole subject. No model, no hands, no extra props. The bag is upright and styled naturally.

Composition: Bag centered in frame with breathing room on all sides. Three-quarter front angle so handles, body, and one side panel are all visible.

Background: Seamless, minimal, lightly textured surface such as linen, matte paper, soft plaster, or warm stone. No furniture, no clutter, no text, no logos.

Lighting: Soft diffused daylight from one side, gentle natural shadow on the opposite side. Highlights catch leather grain, fabric weave, beads, or hardware.

Style: High-end commercial product photography, photorealistic, sharp focus on the product, shallow depth of field, true-to-life color.

Aspect ratio: 1:1 (square, for product grid)."""

BAG_MODEL_PROMPT = """Product picture with model
Using the uploaded product photo as reference, create a high-end ecommerce lifestyle product photograph featuring the same handbag worn by a model.

IMPORTANT:
- The handbag must match the exact design, color, texture, hardware, stitching, and proportions of the uploaded reference photo. Do not alter the product - replicate it faithfully.
- The bag is the primary subject; the model provides scale and styling context.
- Do not add brand names, store names, text, watermarks, labels, or logos anywhere in the image.

Scene: A confident Black woman in her late 20s, shown from chest to mid-thigh. She is holding the bag naturally by the top handle, on her shoulder, or in the crook of her arm depending on the bag style. Prioritize bag visibility.

Wardrobe: Elegant, modern outfit in neutral or complementary tones. Subtle African print accent allowed, but the outfit should not compete with the bag.

Composition: Vertical or square framing. Bag positioned at the visual focal point, roughly center of frame. Three-quarter front angle on the bag.

Background: Seamless, minimal, lightly textured surface. No furniture, no clutter, no text, no logos.

Lighting: Soft diffused daylight from one side, true-to-life color on the bag.

Style: High-end commercial fashion photography, photorealistic, sharp focus on the bag, realistic hands holding the bag correctly.

Aspect ratio: 4:5 (vertical, for product grid and listing pages)."""

BAG_PURSE_MODEL_PROMPT = """Purses prompt
Using the uploaded product photo as reference, create a high-end ecommerce lifestyle product photograph featuring the same small purse or clutch held by a model.

IMPORTANT:
- The purse must match the exact design, color, texture, hardware, stitching, and proportions of the uploaded reference photo. Do not alter the product - replicate it faithfully.
- The purse is the primary subject; the model provides scale and styling context.
- Do not add brand names, store names, text, watermarks, labels, or logos anywhere in the image.

Scene: A confident Black woman in her late 20s, shown from chest to mid-thigh. She is holding the purse naturally, not strapped on her shoulder. The purse must be fully visible and centered in the frame but not oversized.

Wardrobe: Elegant, modern outfit in neutral or complementary tones. Subtle African print accent allowed, but the outfit should not compete with the purse.

Composition: Vertical or square framing. Bag positioned at the visual focal point. Three-quarter front angle on the purse.

Background: Seamless, minimal, lightly textured surface. No furniture, no clutter, no text, no logos.

Lighting: Soft diffused daylight from one side, true-to-life color on the purse.

Style: High-end commercial fashion photography, photorealistic, sharp focus on the purse, realistic hands holding the purse correctly.

Aspect ratio: 4:5 (vertical, for product grid and listing pages)."""

PERFUME_FIDELITY_RULES = """Product fidelity rules:
- If multiple reference photos are provided, they are different angles of the same exact perfume product. Use all of them to understand bottle shape, cap geometry, label placement, liquid color, sprayer, embossing, box shape, and proportions.
- Preserve exact bottle silhouette, glass thickness, cap shape, label shape, label placement, color, material finish, and visible text from the reference. Do not invent brand text or redesign the label.
- Do not change bottle height, shoulder shape, base thickness, cap size, nozzle placement, box dimensions, or label typography.
- Reflections and lighting may improve, but the product design may not change."""

PERFUME_LIFESTYLE_PROMPT = """Product picture with no model
Using the uploaded product photo as reference, create a clean, high-end ecommerce product photograph of the same perfume.

IMPORTANT:
- Preserve the exact bottle, cap, sprayer, label, liquid color, packaging, proportions, and visible text from the reference.
- Do not invent brand names, labels, claims, watermarks, or extra text.

Scene: The perfume bottle or box presented as the sole subject on a minimal premium surface. No model, no hands, no extra props.

Composition: Centered, upright, three-quarter front angle showing bottle shape, cap, label, and side depth.

Background: Seamless, minimal, lightly textured surface such as stone, matte paper, plaster, or glass. Tone complements the fragrance color.

Lighting: Soft diffused studio daylight with controlled highlights through the glass and gentle shadow.

Style: High-end commercial fragrance photography, photorealistic, sharp focus on label and cap, true-to-life color.

Aspect ratio: 1:1 (square, for product grid)."""

PERFUME_MODEL_PROMPT = """Product picture with model
Using the uploaded product photo as reference, create a high-end ecommerce lifestyle photograph featuring the same perfume held by a model.

IMPORTANT:
- Preserve exact bottle shape, cap, label, liquid color, packaging, proportions, and visible text from the reference.
- Do not invent brand names, labels, watermarks, or extra text.

Scene: A confident adult model holds the perfume bottle naturally near chest height or on a vanity-style surface, with the bottle fully visible and label facing camera.

Composition: Vertical framing. Product is the focal point; hands must not cover the label, cap, or bottle shape.

Background: Minimal premium backdrop in warm neutral tones. No clutter.

Lighting: Soft diffused daylight, controlled reflections, true-to-life color.

Style: High-end commercial beauty photography, photorealistic, sharp focus on the perfume, realistic hands.

Aspect ratio: 4:5 (vertical, for product grid and listing pages)."""

GLASSES_FIDELITY_RULES = """Product fidelity rules:
- If multiple reference photos are provided, they are different angles of the same exact eyewear product. Use all of them to understand frame shape, bridge, hinges, temples, lens tint, lens transparency, nose pads, and proportions.
- Preserve exact frame silhouette, lens shape, bridge width, hinge placement, temple thickness, frame color, lens tint, and material finish.
- Do not warp symmetry, bend arms incorrectly, change lens shape, add fake logos, alter tint, or change frame thickness.
- The scene may change, but the eyewear design may not."""

GLASSES_LIFESTYLE_PROMPT = """Product picture with no model
Using the uploaded product photo as reference, create a clean, high-end ecommerce product photograph of the same glasses.

IMPORTANT:
- Preserve exact frame shape, lens shape, bridge, hinges, temple arms, lens tint, material, color, and proportions.
- Do not add brand names, store names, text, watermarks, labels, or logos anywhere in the image.

Scene: The glasses presented as the sole subject on a minimal premium surface. No model, no hands, no extra props.

Composition: Centered, three-quarter front angle showing the frame front, bridge, lenses, and one temple arm.

Background: Seamless, lightly textured neutral surface. No clutter.

Lighting: Soft diffused daylight with controlled lens reflections and true-to-life tint.

Style: High-end commercial eyewear photography, photorealistic, sharp focus, symmetrical frame.

Aspect ratio: 1:1 (square, for product grid)."""

GLASSES_MODEL_PROMPT = """Product picture with model
Using the uploaded product photo as reference, create a high-end ecommerce lifestyle photograph featuring the same glasses worn by a model.

IMPORTANT:
- Preserve exact frame shape, lens shape, bridge, hinges, temple arms, lens tint, material, color, and proportions.
- Do not add brand names, store names, text, watermarks, labels, or logos anywhere in the image.

Scene: A confident adult model wearing the glasses naturally. Face may be cropped, but the glasses must be fully visible, symmetrical, and correctly seated on the face.

Composition: Vertical portrait crop, clean styling, eyewear at the visual focal point.

Background: Minimal premium backdrop in neutral tones. No clutter.

Lighting: Soft diffused daylight with controlled lens reflections and true-to-life tint.

Style: High-end commercial eyewear photography, photorealistic, sharp focus on glasses, natural skin texture.

Aspect ratio: 4:5 (vertical, for product grid and listing pages)."""

BAG_PROMPT_PACK = ProductPromptPack(
    product_type="bag",
    classify_prompt=(
        "Analyze this bag/accessory photo and extract ecommerce attributes. "
        "Identify the product kind, color, material, hardware, closure, shape, "
        "strap/handle details, and style tags."
    ),
    listing_prompt="Write a Shopify product listing for this bag or accessory.",
    fidelity_rules=BAG_FIDELITY_RULES,
    lifestyle_prompt=BAG_LIFESTYLE_PROMPT,
    model_prompt=BAG_MODEL_PROMPT,
    purse_model_prompt=BAG_PURSE_MODEL_PROMPT,
    review_focus=(
        "exact outer silhouette, proportions, handle placement, strap/chain "
        "attachment points, clasp position, hardware, and material texture"
    ),
)

PERFUME_PROMPT_PACK = ProductPromptPack(
    product_type="perfume",
    classify_prompt=(
        "Analyze this perfume/fragrance photo and extract ecommerce attributes. "
        "Identify bottle shape, cap, sprayer, label, liquid color, packaging, "
        "size cues, scent-family clues if visible, and style tags. Do not invent "
        "scent notes that are not visible."
    ),
    listing_prompt="Write a Shopify product listing for this perfume or fragrance product.",
    fidelity_rules=PERFUME_FIDELITY_RULES,
    lifestyle_prompt=PERFUME_LIFESTYLE_PROMPT,
    model_prompt=PERFUME_MODEL_PROMPT,
    review_focus=(
        "exact bottle silhouette, glass shape, cap geometry, sprayer, label "
        "placement, visible text, liquid color, packaging, and proportions"
    ),
)

GLASSES_PROMPT_PACK = ProductPromptPack(
    product_type="glasses",
    classify_prompt=(
        "Analyze this eyewear photo and extract ecommerce attributes. Identify "
        "frame shape, lens shape, lens tint, frame color, material, bridge, "
        "hinges, temples, fit/style cues, and style tags."
    ),
    listing_prompt="Write a Shopify product listing for this eyewear product.",
    fidelity_rules=GLASSES_FIDELITY_RULES,
    lifestyle_prompt=GLASSES_LIFESTYLE_PROMPT,
    model_prompt=GLASSES_MODEL_PROMPT,
    review_focus=(
        "frame symmetry, lens shape, bridge width, hinge placement, temple arms, "
        "frame color, lens tint, material finish, and proportions"
    ),
)

PROMPT_PACKS = {
    "bag": BAG_PROMPT_PACK,
    "perfume": PERFUME_PROMPT_PACK,
    "glasses": GLASSES_PROMPT_PACK,
}


def get_prompt_pack(product_type: str | None) -> ProductPromptPack:
    return PROMPT_PACKS.get(product_type or "", BAG_PROMPT_PACK)
