"""FastAPI entrypoint. One endpoint: POST product photos, get a draft listing.

Run locally:  uvicorn app.main:app --reload
Then open:    http://127.0.0.1:8000/docs  (Swagger UI — upload photos here)
"""

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import pipeline
from app.schemas import ProcessResult

app = FastAPI(title="OHH — product photos to draft listing")

ALLOWED_MEDIA = {"image/jpeg", "image/png", "image/webp"}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/process", response_model=ProcessResult)
async def process(photo: list[UploadFile] = File(...)) -> ProcessResult:
    if not photo:
        raise HTTPException(status_code=400, detail="Upload at least one photo")

    reference_images: list[tuple[bytes, str]] = []
    for upload in photo:
        if upload.content_type not in ALLOWED_MEDIA:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported type {upload.content_type}; "
                    f"use {sorted(ALLOWED_MEDIA)}"
                ),
            )

        image_bytes = await upload.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail=f"Empty file: {upload.filename}")
        reference_images.append((image_bytes, upload.content_type))

    # TODO(chris): for a single-user internal tool this synchronous flow is fine
    # (it'll block ~a few seconds on the Claude calls). If it gets slow, move to
    # a background task + polling, but don't reach for that until you feel it.
    primary_bytes, primary_media_type = reference_images[0]
    return await pipeline.run(
        primary_bytes,
        primary_media_type,
        reference_images=reference_images,
    )
