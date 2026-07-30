"""Local drag-and-drop grouping studio.

The ONE step we take away from the AI: deciding which photos are the same
physical product. You drag images from the unsorted pile into product bins,
tag color variants, and Save — which writes the exact folder structure
`app.batch` already consumes:  incoming/<category>/<product>/<variant>/IMG_*.jpg

Run locally:
    uvicorn app.grouper:app --reload
Then open:
    http://127.0.0.1:8000/

Nothing here calls any paid API. Grouping is purely local file organization;
the pipeline (naming, background removal, copy, image gen) runs afterward.
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel

from app.batch import IMAGE_EXTENSIONS, _is_product_dir
from app.config import settings

app = FastAPI(title="Grouping studio")

# Everything the tool touches lives under this root. Product folders are direct
# children of incoming/<category>; the loose pile to sort is incoming/<category>/_unsorted.
INCOMING = Path("incoming")
UNSORTED_DIRNAME = "_unsorted"
THUMB_MAX = 320  # px, longest edge for the grid thumbnails
_STATIC = Path(__file__).parent / "static"


# --------------------------------------------------------------------------- #
# Path safety — never read/write outside incoming/. Reuse for EVERY path that
# comes from the browser; a request could otherwise pass "../../etc/passwd".
# --------------------------------------------------------------------------- #
def _safe(rel: str) -> Path:
    """Resolve a browser-supplied relative path, confined to INCOMING.

    Raises 400 if the resolved path escapes INCOMING. Returns an absolute Path.
    """
    root = INCOMING.resolve()
    target = (root / rel).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=400, detail=f"path escapes incoming/: {rel}")
    return target


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _safe_folder_name(name: str) -> str:
    """Make a product/variant name usable as a folder without losing readability.

    The pipeline reads the LABEL for the real Shopify title, so the folder is
    just a container — we only need it filesystem-safe and non-empty. Keeps
    spaces, apostrophes, hyphens, & etc. (the existing catalog uses those);
    strips only path separators and control characters.
    """
    cleaned = name.replace("/", "-").replace("\\", "-")
    cleaned = re.sub(r"[\x00-\x1f]", "", cleaned)
    cleaned = cleaned.strip().strip(".")  # no leading/trailing dots or spaces
    return cleaned or "Untitled"


# --------------------------------------------------------------------------- #
# Request/response models. `rel` fields are paths relative to INCOMING so the
# browser never sees absolute filesystem paths.
# --------------------------------------------------------------------------- #
class ImageInfo(BaseModel):
    rel: str          # e.g. "perfumes/_unsorted/IMG_3241.jpeg"
    name: str         # e.g. "IMG_3241"
    number: int | None  # parsed IMG number for adjacency, or None


class VariantSpec(BaseModel):
    name: str            # "" / "Default" -> product-level (no variant subfolder)
    images: list[str]    # list of `rel` paths from the unsorted pile


class ProductSpec(BaseModel):
    name: str            # product folder name, e.g. "Bade'e Al Oud Sublime"
    variants: list[VariantSpec]


class SaveRequest(BaseModel):
    category: str            # e.g. "perfumes"
    products: list[ProductSpec]
    unsorted: list[str] = []  # rels that should sit loose in <category>/_unsorted


class ProductGroup(BaseModel):
    name: str
    variants: list[dict]     # [{"name": str, "images": [ImageInfo-as-dict]}]


class SaveResult(BaseModel):
    moved: int
    products_created: int
    undo_log: str
    errors: list[str]


# --------------------------------------------------------------------------- #
# Static page
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = _STATIC / "grouper.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="static/grouper.html missing")
    return HTMLResponse(page.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Read endpoints — these are fully implemented; they just enumerate the pile.
# --------------------------------------------------------------------------- #
@app.get("/api/categories")
def categories() -> dict:
    """Category folders under incoming/ plus how many loose images each has."""
    if not INCOMING.exists():
        return {"categories": []}
    out = []
    for child in sorted(INCOMING.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        pile = child / UNSORTED_DIRNAME
        count = sum(1 for p in pile.glob("*") if _is_image(p)) if pile.exists() else 0
        out.append({"category": child.name, "unsorted": count})
    return {"categories": out}


def _image_infos(folder: Path) -> list[ImageInfo]:
    """Number-sorted ImageInfo for the images directly inside a folder."""
    infos = [
        ImageInfo(
            rel=str(p.relative_to(INCOMING.resolve())),
            name=p.stem,
            number=_img_number(p),
        )
        for p in folder.iterdir()
        if _is_image(p)
    ]
    infos.sort(key=lambda i: (i.number is None, i.number or 0, i.name))
    return infos


@app.get("/api/unsorted", response_model=list[ImageInfo])
def unsorted(category: str = Query(...)) -> list[ImageInfo]:
    """Every loose image in incoming/<category>/_unsorted, number-sorted."""
    pile = _safe(f"{category}/{UNSORTED_DIRNAME}")
    return _image_infos(pile) if pile.exists() else []


@app.get("/api/products")
def products(category: str = Query(...)) -> dict:
    """Existing product folders in a category with their color variants.

    Feeds the review view: each product folder -> a bin; each color subfolder ->
    a named color row; images sitting directly in the product folder -> one
    unnamed color row. Skips _-prefixed buckets (display boards, trays, ...).
    """
    base = _safe(category)
    if not base.exists():
        return {"products": []}
    out: list[dict] = []
    for prod in sorted(p for p in base.iterdir() if _is_product_dir(p)):
        variants: list[dict] = []
        flat = _image_infos(prod)
        if flat:  # flat images = an unnamed color (shown first)
            variants.append({"name": "", "images": [i.model_dump() for i in flat]})
        for sub in sorted(d for d in prod.iterdir() if _is_product_dir(d)):
            imgs = _image_infos(sub)
            if imgs:
                variants.append({"name": sub.name, "images": [i.model_dump() for i in imgs]})
        if variants:
            out.append({"name": prod.name, "variants": variants})
    return {"products": out}


@app.get("/api/thumb")
def thumb(rel: str = Query(...)) -> Response:
    """Downscaled JPEG thumbnail for the grid (keeps the page light for 1000+)."""
    path = _safe(rel)
    if not _is_image(path):
        raise HTTPException(status_code=404, detail="not an image")
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        im.thumbnail((THUMB_MAX, THUMB_MAX))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as exc:  # noqa: BLE001 - surface as 500 for the UI
        raise HTTPException(status_code=500, detail=f"thumb failed: {exc}") from exc


@app.get("/api/full")
def full(rel: str = Query(...)) -> FileResponse:
    """Original image, for the hover/zoom-to-read-the-label view."""
    path = _safe(rel)
    if not _is_image(path):
        raise HTTPException(status_code=404, detail="not an image")
    return FileResponse(path)


# --------------------------------------------------------------------------- #
# Helper: parse the IMG number (e.g. IMG_3241 -> 3241). Used by adjacency.
# --------------------------------------------------------------------------- #
def _img_number(path: Path) -> int | None:
    """Trailing integer of a stem (IMG_3241 -> 3241); None if there's none.

    Plumbing for the pile view + adjacency. Implemented so the grid renders
    out of the box; the interesting logic below is yours.
    """
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


# --------------------------------------------------------------------------- #
# CORE LOGIC #1 — adjacency pre-grouping (the "assist", not the decider).
# You shoot all of one product's variants together, so IMG numbers cluster.
# Group the unsorted pile into DRAFT bins by number gaps; the user then merges/
# splits. This turns "build 150 groups from scratch" into "correct ~30".
# --------------------------------------------------------------------------- #
@app.get("/api/suggest")
def suggest(
    category: str = Query(...),
    gap: int = Query(8, description="a jump larger than this starts a new group"),
) -> dict:
    """Return draft product bins for the unsorted pile, keyed by adjacency.

    Gap-based clustering: images arrive number-sorted, so a run of near-
    consecutive IMG numbers is almost always one product's variants shot in a
    row. We keep extending the current bin while each number is within `gap` of
    the previous one; a bigger jump (or a number-less file) starts a new bin.
    A number-less image can't anchor adjacency, so it lands in its own bin.

    Purely a suggestion — no files move here. The UI renders these as pre-filled
    bins the user then merges/splits.
    """
    images = unsorted(category)  # already number-sorted ImageInfo list

    groups: list[list[str]] = []
    current: list[str] = []
    prev: int | None = None
    for info in images:
        number = info.number
        contiguous = (
            current
            and number is not None
            and prev is not None
            and abs(number - prev) <= gap
        )
        if contiguous:
            current.append(info.rel)
        else:
            if current:
                groups.append(current)
            current = [info.rel]
        prev = number
    if current:
        groups.append(current)

    return {"groups": groups}


# --------------------------------------------------------------------------- #
# CORE LOGIC #2 — apply the groupings: move files into product/variant folders.
# This is the payoff step. It must be safe (confine to INCOMING), idempotent
# enough to re-run, and reversible (write an undo log to outputs/, NOT /tmp —
# a reboot wiped /tmp on us before).
# --------------------------------------------------------------------------- #
def _cleanup_empty(cat_dir: Path, pile: Path) -> None:
    """Remove product/color folders emptied by a save; keep root + _unsorted."""
    for root, _dirs, _files in os.walk(cat_dir, topdown=False):
        p = Path(root)
        if p == cat_dir or p == pile:
            continue
        remaining = [x for x in p.iterdir() if x.name != ".DS_Store"]
        if not remaining:
            for junk in p.iterdir():
                junk.unlink()          # clear a lone .DS_Store
            try:
                p.rmdir()
            except OSError:
                pass


@app.post("/api/save", response_model=SaveResult)
def save(req: SaveRequest) -> SaveResult:
    """Reconcile a category to the layout in the request.

    Works for both flows: sorting the _unsorted pile AND reviewing/correcting an
    already-grouped category (moving photos between products/colors, spinning
    off new products, or dropping some back to _unsorted). Every source must be
    inside this category; images already in the right place are left untouched;
    nothing is overwritten; emptied folders are cleaned; and a reversible undo
    log is written to outputs/ (never /tmp). Per-file problems collect into
    `errors` rather than aborting the whole save.
    """
    cat_dir = _safe(req.category)
    pile = _safe(f"{req.category}/{UNSORTED_DIRNAME}")
    undo: list[dict] = []
    errors: list[str] = []
    products_created = 0

    def place(rel: str, dest_dir: Path) -> bool:
        """Move rel into dest_dir (or leave it if already there). True if placed."""
        try:
            src = _safe(rel)
        except HTTPException as exc:
            errors.append(f"{rel}: {exc.detail}")
            return False
        if not _is_image(src):
            errors.append(f"{rel}: not an existing image")
            return False
        if cat_dir != src and cat_dir not in src.parents:
            errors.append(f"{rel}: outside category {req.category}")
            return False
        dest_file = dest_dir / src.name
        if src == dest_file:
            return True                       # already in the right place
        if dest_file.exists():
            errors.append(f"{rel}: {dest_file.name} already exists in target")
            return False
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            os.rename(src, dest_file)
        except OSError as exc:
            errors.append(f"{rel}: move failed: {exc}")
            return False
        undo.append({"from": str(dest_file), "to": str(src)})
        return True

    for product in req.products:
        # Every image belongs to a color. A product may be a single unnamed
        # color (flat) OR several named colors (subfolders) — never both: an
        # unnamed group beside named ones would land at the product root, which
        # app.batch drops when color subfolders exist. Skip rather than lose it.
        is_root = lambda v: v.name.strip().lower() in ("", "default") and v.images
        has_named = any(not is_root(v) and v.images for v in product.variants)
        if has_named and any(is_root(v) for v in product.variants):
            errors.append(
                f"{product.name}: has an unnamed color group alongside named "
                f"colors; name every color (skipped to avoid dropped images)"
            )
            continue

        dest_base = _safe(f"{req.category}/{_safe_folder_name(product.name)}")
        placed = 0
        for variant in product.variants:
            if variant.name.strip().lower() in ("", "default"):
                dest_dir = dest_base
            else:
                dest_dir = dest_base / _safe_folder_name(variant.name)
            for rel in variant.images:
                if place(rel, dest_dir):
                    placed += 1
        if placed:
            products_created += 1

    # images dragged back to the pile
    for rel in req.unsorted:
        place(rel, pile)

    _cleanup_empty(cat_dir, pile)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    undo_path = Path(settings.output_dir) / f"grouper_undo_{stamp}.jsonl"
    if undo:
        undo_path.parent.mkdir(parents=True, exist_ok=True)
        undo_path.write_text(
            "\n".join(json.dumps(entry) for entry in undo) + "\n", encoding="utf-8"
        )

    return SaveResult(
        moved=len(undo),
        products_created=products_created,
        undo_log=str(undo_path) if undo else "",
        errors=errors,
    )
