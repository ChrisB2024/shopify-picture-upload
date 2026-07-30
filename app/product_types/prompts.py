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

Scene: A confident Black woman in her late 20s holds the perfume bottle naturally near chest height or on a vanity-style surface, with the bottle fully visible and label facing camera.

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

JEWELRY_FIDELITY_RULES = """Product fidelity rules:
- If multiple reference photos are provided, they are different angles of the same exact jewelry piece. Use all of them to understand the metal tone, stone layout, chain link, pendant shape, clasp, and proportions.
- Preserve the exact metal color and finish (yellow gold, rose gold, silver, gunmetal), stone color, cut, count and arrangement, chain link type and thickness, pendant/charm shape, bezel/prong settings, and clasp.
- Do not change the link pattern, stone layout, pendant proportions, or metal tone. Do not add fake hallmarks, engravings, brand names, or logos.
- Reflections and sparkle may improve, but the design may not change."""

JEWELRY_LIFESTYLE_PROMPT = """Product picture with no model
Using the uploaded product photo as reference, create a clean, high-end ecommerce product photograph of the same jewelry piece.

IMPORTANT:
- Preserve the exact metal tone and finish, stone color/cut/arrangement, chain link, pendant shape, setting, and proportions from the reference. Replicate it faithfully.
- Do not add brand names, store names, text, watermarks, hallmarks, or logos anywhere in the image.

Scene: The jewelry presented on its own as the sole subject, elegantly arranged - a necklace gently curved or coiled, a bracelet or bangle in a soft loop, earrings shown as a clean pair, a ring standing upright. No model, no hands, no clutter.

Composition: Piece centered and filling the frame with breathing room. Angle chosen so links, stones, and clasp read clearly.

Background: Seamless premium surface such as soft velvet, matte marble, satin, or warm stone. Tone complements the metal. No clutter, no text, no logos.

Lighting: Soft diffused studio light with controlled sparkle and clean reflections on metal and stones. True-to-life metal tone and gem color.

Style: High-end commercial jewelry photography, photorealistic, macro-sharp on stones and links, shallow depth of field.

Aspect ratio: 1:1 (square, for product grid)."""

JEWELRY_MODEL_PROMPT = """Product picture with model
Using the uploaded product photo as reference, create a high-end ecommerce lifestyle photograph featuring the same jewelry piece worn by a model.

IMPORTANT:
- Preserve the exact metal tone, stone color/cut/arrangement, chain link, pendant shape, setting, and proportions from the reference. Do not alter the piece.
- Do not add brand names, store names, text, watermarks, hallmarks, or logos anywhere in the image.

Scene: A confident Black woman in her late 20s wearing the piece the way it is meant to be worn - a necklace or chain on the neckline/décolletage, a bracelet, bangle, or cuff on the wrist, earrings on the ear in a soft side profile, a ring on the finger. Crop to the relevant area so the jewelry is the clear focus and correctly worn.

Wardrobe: Elegant, modern, neutral or complementary tones with clean skin. Subtle African print accent allowed, but nothing that competes with the piece.

Composition: Vertical framing. The jewelry sits at the visual focal point, in sharp focus, fully visible and not obscured by hair or fabric.

Background: Minimal premium backdrop in warm neutral tones. No clutter, no text, no logos.

Lighting: Soft diffused daylight with controlled sparkle, natural skin texture, true-to-life metal and gem color.

Style: High-end commercial jewelry/beauty photography, photorealistic, sharp focus on the piece.

Aspect ratio: 4:5 (vertical, for product grid and listing pages)."""

LOTION_FIDELITY_RULES = """Product fidelity rules:
- If multiple reference photos are provided, they are different angles of the same exact product. Use all of them to understand the container shape (bottle, tube, jar, pump), cap, label placement, color, and proportions.
- Preserve the exact container silhouette, cap/pump shape, label shape and placement, color, material finish, and visible text from the reference. Do not invent brand text, claims, or redesign the label.
- Do not change container height, shoulder, cap size, pump geometry, or label typography.
- Reflections and lighting may improve, but the product design may not change."""

LOTION_LIFESTYLE_PROMPT = """Product picture with no model
Using the uploaded product photo as reference, create a clean, high-end ecommerce product photograph of the same skincare/haircare product.

IMPORTANT:
- Preserve the exact container shape, cap/pump, label, color, packaging, proportions, and visible text from the reference.
- Do not invent brand names, labels, claims, watermarks, or extra text.

Scene: The product presented as the sole subject on a clean spa or vanity surface. At most one subtle prop that complements the product (a folded towel, a sprig of greenery, a smooth stone). No clutter, no hands, no model.

Composition: Centered, upright, three-quarter front angle showing the container shape, cap, and label clearly.

Background: Seamless minimal surface such as matte stone, plaster, or warm neutral tile. Tone complements the packaging.

Lighting: Soft diffused studio daylight, clean highlights, gentle shadow, true-to-life color.

Style: High-end commercial skincare photography, photorealistic, sharp focus on the label and cap.

Aspect ratio: 1:1 (square, for product grid)."""

LOTION_MODEL_PROMPT = """Product picture with model
Using the uploaded product photo as reference, create a high-end ecommerce lifestyle photograph featuring the same product held by a model.

IMPORTANT:
- Preserve the exact container shape, cap/pump, label, color, packaging, proportions, and visible text from the reference.
- Do not invent brand names, labels, watermarks, or extra text.

Scene: A confident Black woman in her late 20s holds the product naturally near chest height, or presents it against smooth, healthy-looking skin, with the label facing camera and fully visible. Hands must not cover the label or container shape.

Wardrobe: Clean, modern, neutral tones. Subtle African print accent allowed, but nothing that competes with the product.

Composition: Vertical framing. Product is the focal point and in sharp focus.

Background: Minimal premium backdrop in warm neutral tones. No clutter, no text, no logos.

Lighting: Soft diffused daylight, natural skin texture, true-to-life color.

Style: High-end commercial beauty photography, photorealistic, realistic hands, sharp focus on the product.

Aspect ratio: 4:5 (vertical, for product grid and listing pages)."""

CLIPPER_FIDELITY_RULES = """Product fidelity rules:
- If multiple reference photos are provided, they are different angles of the same exact grooming tool. Use all of them to understand the body shape, blade/head, guards and attachments, buttons, cord or cordless design, color, and proportions.
- Preserve the exact device silhouette, blade/head shape, attachment guards, button and dial placement, color, material finish, and any visible branding exactly as shown. Do not invent brand text or redesign the device.
- Do not change the body proportions, blade geometry, guard shapes, or control placement.
- Reflections and lighting may improve, but the product design may not change."""

CLIPPER_LIFESTYLE_PROMPT = """Product picture with no model
Using the uploaded product photo as reference, create a clean, high-end ecommerce product photograph of the same grooming tool.

IMPORTANT:
- Preserve the exact device shape, blade/head, guards and attachments, controls, color, and any visible branding from the reference.
- Do not invent brand names, labels, watermarks, or extra text.

Scene: The device presented as the main subject on a clean barber/grooming surface. Its guard combs or attachments may be neatly arranged beside it if they appear in the reference. No hands, no model, no clutter.

Composition: Centered, three-quarter angle showing the body, blade/head, and controls clearly.

Background: Seamless minimal surface such as matte stone, brushed metal, or warm neutral. No clutter, no text.

Lighting: Soft diffused studio light with controlled highlights on the housing and blade, true-to-life color.

Style: High-end commercial product photography, photorealistic, sharp focus on the device.

Aspect ratio: 1:1 (square, for product grid)."""

CLIPPER_MODEL_PROMPT = """Product picture with model
Using the uploaded product photo as reference, create a high-end ecommerce lifestyle photograph featuring the same grooming tool in use.

IMPORTANT:
- Preserve the exact device shape, blade/head, guards, controls, color, and any visible branding from the reference. Do not alter the device.
- Do not invent brand names, labels, watermarks, or extra text.

Scene: A well-groomed Black man in his late 20s or 30s holding and using the grooming tool naturally (trimming his beard or hairline), with the device clearly visible in hand and correctly oriented. The device is the primary subject.

Wardrobe: Clean, modern, neutral tones in a tidy barbershop-style setting.

Composition: Vertical framing. The device is the focal point and in sharp focus; the hand grips it realistically.

Background: Minimal premium backdrop in neutral tones. No clutter, no text, no logos.

Lighting: Soft diffused daylight, natural skin texture, true-to-life color on the device.

Style: High-end commercial grooming photography, photorealistic, realistic hands, sharp focus on the device.

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

JEWELRY_PROMPT_PACK = ProductPromptPack(
    product_type="jewelry",
    classify_prompt=(
        "Analyze this jewelry photo and extract ecommerce attributes. Identify the "
        "piece type (necklace, chain, bracelet, bangle, cuff, earrings, ring, "
        "pendant), metal tone, stone color/cut/count, chain link, pendant/charm, "
        "clasp/setting, and style tags. Do not invent hallmarks or materials that "
        "are not visible."
    ),
    listing_prompt="Write a Shopify product listing for this jewelry piece.",
    fidelity_rules=JEWELRY_FIDELITY_RULES,
    lifestyle_prompt=JEWELRY_LIFESTYLE_PROMPT,
    model_prompt=JEWELRY_MODEL_PROMPT,
    review_focus=(
        "exact metal tone and finish, stone color/cut/arrangement, chain link "
        "type, pendant shape, setting, clasp, and proportions"
    ),
)

LOTION_PROMPT_PACK = ProductPromptPack(
    product_type="lotion",
    classify_prompt=(
        "Analyze this skincare/haircare product photo and extract ecommerce "
        "attributes. Identify the container type (bottle, tube, jar, pump), cap, "
        "label, color, size cues, product type if visible (lotion, cream, oil, "
        "butter), and style tags. Do not invent ingredients or claims that are "
        "not visible."
    ),
    listing_prompt="Write a Shopify product listing for this skincare or haircare product.",
    fidelity_rules=LOTION_FIDELITY_RULES,
    lifestyle_prompt=LOTION_LIFESTYLE_PROMPT,
    model_prompt=LOTION_MODEL_PROMPT,
    review_focus=(
        "exact container silhouette, cap/pump geometry, label placement, visible "
        "text, color, packaging, and proportions"
    ),
)

CLIPPER_PROMPT_PACK = ProductPromptPack(
    product_type="clipper",
    classify_prompt=(
        "Analyze this grooming-tool photo and extract ecommerce attributes. "
        "Identify the tool type (hair clipper, beard trimmer, nose trimmer, "
        "shaver), body shape, blade/head, guards and attachments, controls, "
        "cord/cordless, color, visible branding, and style tags."
    ),
    listing_prompt="Write a Shopify product listing for this grooming tool.",
    fidelity_rules=CLIPPER_FIDELITY_RULES,
    lifestyle_prompt=CLIPPER_LIFESTYLE_PROMPT,
    model_prompt=CLIPPER_MODEL_PROMPT,
    review_focus=(
        "exact device silhouette, blade/head shape, guard attachments, control "
        "placement, color, visible branding, and proportions"
    ),
)

PROMPT_PACKS = {
    "bag": BAG_PROMPT_PACK,
    "perfume": PERFUME_PROMPT_PACK,
    "glasses": GLASSES_PROMPT_PACK,
    "jewelry": JEWELRY_PROMPT_PACK,
    "lotion": LOTION_PROMPT_PACK,
    "clipper": CLIPPER_PROMPT_PACK,
}


def get_prompt_pack(product_type: str | None) -> ProductPromptPack:
    return PROMPT_PACKS.get(product_type or "", BAG_PROMPT_PACK)
