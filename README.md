# Pet Product Preview Backend

Free local FastAPI MVP for composing a user-uploaded pet product image onto a user-uploaded dog photo. The backend keeps the original simple overlay mode and adds an experimental dog harness fitting pipeline that runs locally with open-source Python packages.

No paid APIs, database, Flutter code, or OpenAI image generation are used.

## What This Does

- Upload a dog photo and product image.
- Remove the product image background locally with `rembg`.
- Save uploaded files under `static/uploads`.
- Save generated preview images under `static/results`.
- Optionally save fitting debug images under `static/debug`.
- Return the result image path, image size, mode, fixed prompt text, and fitting metadata when available.

## Modes

`overlay` mode is the original sticker-like composition. The transparent product is optionally scaled, placed at `x`/`y`, opacity-adjusted, shadowed, and alpha-composited directly over the dog photo. It does not segment the dog, estimate keypoints, warp the product, or restore dog pixels over the product.

`fit_harness` mode is experimental. It segments the dog, estimates coarse body landmarks from the mask, perspective-warps the transparent product toward the torso, creates a soft occlusion mask around neck/chest/shoulder areas, applies lightweight edge refinement, and composites the result. The product texture, color, logo, and visual identity are preserved as much as possible, but the harness may be warped because fitting requires shape adaptation.

## Backend Folder Structure

```text
app/
  main.py
  api/
    preview.py
  services/
    background_remove_service.py
    composite_service.py
    dog_segmentation_service.py
    dog_keypoint_service.py
    product_warp_service.py
    occlusion_service.py
    inpainting_refine_service.py
    prompt_service.py
  prompts/
    pet_product_preview.txt
static/
  uploads/
  results/
  debug/
requirements.txt
```

## Requirements

- Python 3.10 or newer is recommended.
- Do not commit a virtual environment folder.
- Internet access may be needed the first time `rembg` downloads its local model file.
- After models are cached locally, the API runs without paid external services.

Install packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

The API will run at `http://127.0.0.1:8000`, with Swagger UI at `http://127.0.0.1:8000/docs`.

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
  "x": 330,
  "y": 520,
  "scale": 0.42,
  "opacity": 0.95,
  "mode": "overlay",
  "auto_fit": true,
  "warp_strength": 0.75,
  "occlusion_strength": 0.45,
  "debug": false
}
```

Placement rules:

- `mode`: optional. Use `"overlay"` or `"fit_harness"`. Missing mode defaults to `"overlay"`.
- `x`: optional product left position in pixels for overlay/manual placement.
- `y`: optional product top position in pixels for overlay/manual placement.
- `scale`: optional product scale multiplier. Overlay mode scales only when this value is explicitly provided.
- `opacity`: optional product opacity from `0` to `1`.
- `auto_fit`: optional boolean for `fit_harness`; defaults to true.
- `warp_strength`: optional fitting warp amount from `0` to `1`; defaults to `0.75`.
- `occlusion_strength`: optional dog-over-harness restoration amount from `0` to `1`; defaults to `0.45`.
- `debug`: optional boolean. When true in `fit_harness`, debug image paths are returned.

## Overlay Example

```bash
curl -X POST http://127.0.0.1:8000/preview/composite \
  -F "dog_image=@/absolute/path/to/dog.jpg" \
  -F "product_image=@/absolute/path/to/harness.png" \
  -F 'placement={"x":330,"y":520,"scale":0.42,"opacity":0.95,"mode":"overlay"}'
```

## Fit Harness Example

```bash
curl -X POST http://127.0.0.1:8000/preview/composite \
  -F "dog_image=@/absolute/path/to/dog.jpg" \
  -F "product_image=@/absolute/path/to/harness.png" \
  -F 'placement={"mode":"fit_harness","auto_fit":true,"warp_strength":0.75,"occlusion_strength":0.45,"debug":true}'
```

Without placement data, the endpoint uses overlay mode and centers the product:

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
  "mode": "fit_harness",
  "keypoints": {
    "body_center": [520, 430],
    "neck_center": [500, 250],
    "chest_center": [515, 380],
    "shoulder_left": [390, 330],
    "shoulder_right": [640, 330],
    "back_center": [510, 290],
    "torso_left": [405, 470],
    "torso_right": [650, 470]
  },
  "debug_paths": {
    "dog_mask_image": "/static/debug/run/dog_mask.png",
    "keypoints_visualization": "/static/debug/run/keypoints.png",
    "warped_product_image": "/static/debug/run/warped_product.png",
    "occlusion_mask": "/static/debug/run/occlusion_mask.png",
    "final_result": "/static/debug/run/final_result.png"
  },
  "fixed_prompt": "Composite the product image onto the dog photo as a pet shopping preview.\n\n..."
}
```

Open returned paths in a browser, for example:

```text
http://127.0.0.1:8000/static/results/pet_product_preview_abc123.png
```

## Fallback Behavior

If `fit_harness` cannot estimate a useful dog segmentation and `auto_fit` is true, the API falls back to `overlay` mode and returns a normal overlay response. If `auto_fit` is false, the API returns a useful `422` error so the caller can ask for a clearer dog image or switch to overlay mode.

## Limitations

- This is not true virtual try-on.
- Dog pose and product angle heavily affect quality.
- Harness fitting is experimental and uses heuristic landmarks, not trained animal pose estimation.
- Product may still look unrealistic without a trained pet virtual try-on model.
- The dog face is not intentionally altered, and the background is not intentionally changed except near the fitted harness blend area.
- No heavy SDXL or diffusion inpainting model is downloaded automatically.

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
