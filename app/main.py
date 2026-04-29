"""FastAPI entry point for local Stable Diffusion dog harness previews."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.generate import router as generate_router
from app.config import RESULT_DIR, STATIC_DIR, UPLOAD_DIR


for directory in (STATIC_DIR, UPLOAD_DIR, RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Dog Harness Stable Diffusion Preview API",
    description="Local FastAPI backend that generates a dog wearing a harness with Stable Diffusion.",
    version="0.3.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(generate_router)


@app.get("/")
async def health_check() -> dict[str, str]:
    """Return a tiny health response for local smoke testing."""

    return {"status": "ok", "service": "dog-harness-stable-diffusion-preview"}


# Example: run with `uvicorn app.main:app --reload`.

