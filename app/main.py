"""FastAPI application entry point for the pet product preview MVP.

Process overview:
1. Create the upload and result folders used by the local static file server.
2. Build the FastAPI app object.
3. Mount the static directory so saved images can be viewed by path.
4. Register the preview router that owns the image-composition endpoint.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.preview import router as preview_router


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"

# Create local folders at startup import time so the MVP works from a fresh clone.
for directory in (STATIC_DIR, UPLOAD_DIR, RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Pet Product Preview API",
    description="Local FastAPI MVP for composing pet product previews without paid APIs.",
    version="0.1.0",
)

# Serve uploaded and generated images from /static/uploads and /static/results.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Attach the preview routes under /preview.
app.include_router(preview_router)


@app.get("/")
async def health_check() -> dict[str, str]:
    """Return a tiny health response for local smoke testing."""

    return {"status": "ok", "service": "pet-product-preview"}


# Example: run with `uvicorn app.main:app --reload`.

