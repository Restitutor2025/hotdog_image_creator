"""FastAPI application entry point for local pet product previews."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.generate import router as generate_router
from app.api.preview import router as preview_router
from app.config import DEBUG_DIR, RESULT_DIR, STATIC_DIR, UPLOAD_DIR


# Create local folders at startup import time so the MVP works from a fresh clone.
for directory in (STATIC_DIR, UPLOAD_DIR, RESULT_DIR, DEBUG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Pet Product Preview API",
    description="Local FastAPI backend for dog clothes and harness previews without paid APIs.",
    version="0.2.0",
)

# Serve uploaded and generated images from /static/uploads and /static/results.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Attach the primary generation route and the legacy compositing route under /preview.
app.include_router(generate_router)
app.include_router(preview_router)


@app.get("/")
async def health_check() -> dict[str, str]:
    """Return a tiny health response for local smoke testing."""

    return {"status": "ok", "service": "pet-product-preview"}


# Example: run with `uvicorn app.main:app --reload`.

