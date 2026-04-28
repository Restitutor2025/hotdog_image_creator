# Pet Product Preview Backend

Free local FastAPI MVP for composing a user-uploaded pet product image onto a user-uploaded dog photo. The implementation uses local Python packages only: `rembg` removes the product background and `Pillow` composites the unchanged product pixels onto the dog image with optional placement and a soft contact shadow.

## What This Does

- Upload a dog photo.
- Upload a product image.
- Remove the product image background locally with `rembg`.
- Composite the product onto the dog photo with `Pillow`.
- Preserve the product image angle, aspect ratio, shape, color, logo, and texture.
- Save uploaded files under `static/uploads`.
- Save generated preview images under `static/results`.
- Return the result image path, image size, and fixed prompt text.

No paid APIs, database, Flutter code, or OpenAI image generation are used.

## Backend Folder Structure

```text
app/
  main.py
  api/
    preview.py
  services/
    background_remove_service.py
    composite_service.py
    prompt_service.py
  prompts/
    pet_product_preview.txt
static/
  uploads/
  results/
requirements.txt
```

## Requirements

- Python 3.10 or newer is recommended.
- Do not commit a virtual environment folder. Each developer should create their own local environment and install packages from `requirements.txt`.
- Internet access is needed the first time `rembg` downloads its local model file.
- After the model is cached locally, the API runs without paid external services.

## Quick Start

From the project root, create a local virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then start the API server:

```bash
uvicorn app.main:app --reload
```

On Windows, activate the virtual environment with:

```bash
.venv\Scripts\activate
```

Then install packages and run the server:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Swagger UI is available at:

```text
http://127.0.0.1:8000/docs
```

## Run Command

Use this after activating your virtual environment and installing `requirements.txt`:

```bash
uvicorn app.main:app --reload
```

## Health Check

```bash
curl http://127.0.0.1:8000/
```

Expected response:

```json
{
  "status": "ok",
  "service": "pet-product-preview"
}
```

## Endpoint

`POST /preview/composite`

Multipart form fields:

- `dog_image`: uploaded dog photo file. Supported common formats include PNG, JPG, JPEG, and WEBP.
- `product_image`: uploaded product image file. Supported common formats include PNG, JPG, JPEG, and WEBP.
- `placement`: optional JSON string.

Placement schema:

```json
{
  "x": 120,
  "y": 180,
  "scale": 0.8,
  "opacity": 0.95
}
```

Placement rules:

- `x`: optional product left position in pixels.
- `y`: optional product top position in pixels.
- `scale`: optional product scale multiplier. The product is scaled only when this value is explicitly provided.
- `opacity`: optional product opacity from `0` to `1`.
- If `x` and `y` are omitted, the product is centered on the dog image.
- If `scale` is omitted, the product keeps its original pixel size.

## Example Request

Replace the image paths with files on your machine:

```bash
curl -X POST http://127.0.0.1:8000/preview/composite \
  -F "dog_image=@/absolute/path/to/dog.jpg" \
  -F "product_image=@/absolute/path/to/product.png" \
  -F 'placement={"x":120,"y":180,"scale":0.8,"opacity":0.95}'
```

Without placement data:

```bash
curl -X POST http://127.0.0.1:8000/preview/composite \
  -F "dog_image=@/absolute/path/to/dog.jpg" \
  -F "product_image=@/absolute/path/to/product.png"
```

## Example Response

```json
{
  "result_image_path": "/static/results/pet_product_preview_abc123.png",
  "width": 1024,
  "height": 768,
  "fixed_prompt": "Composite the product image onto the dog photo as a pet shopping preview.\n\nPreserve the product image exactly:\n..."
}
```

Open the result image in a browser:

```text
http://127.0.0.1:8000/static/results/pet_product_preview_abc123.png
```

## File Storage

- Uploaded dog and product images are saved to `static/uploads`.
- Composited result images are saved to `static/results`.
- Generated upload/result images are ignored by `.gitignore`.
- `.gitkeep` files keep the empty folders available in version control.

## Prompt File

The fixed prompt is stored in:

```text
app/prompts/pet_product_preview.txt
```

The API loads the prompt through:

```text
app/services/prompt_service.py
```

The prompt text is returned in the API response for debugging and logging.

## Notes

- The first composite request can take longer because `rembg` may download its model file.
- The product is not rotated, redrawn, warped, or distorted.
- A simple soft shadow is added under the product.
- The dog photo is used as the base image and is not edited except where the product and local blending are composited.

## Troubleshooting

If multipart upload fails, make sure dependencies are installed:

```bash
pip install -r requirements.txt
```

If `rembg` or `numba` has cache issues, this project creates a local cache folder at:

```text
.cache/numba
```

If port `8000` is already in use, run on another port:

```bash
uvicorn app.main:app --reload --port 8001
```
