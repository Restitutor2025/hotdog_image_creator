# Dog Clothes and Harness Preview Backend

This project is moving from local image compositing to local image generation/editing for dog clothes and dog harness previews. The primary goal is no longer to paste a transparent product image onto a dog photo, but to use a local GPU workflow that behaves more like: "Generate an image of this dog wearing this harness or clothing."

The backend remains FastAPI, free, local, and API-only. It does not use paid APIs, OpenAI image generation, Flutter code, or database code.

## Current Direction

The new primary endpoint is:

```text
POST /preview/generate
```

It accepts a dog photo, a product image, and `product_type` of `harness` or `clothes`, then sends them to a pluggable local image generation backend.

ComfyUI is the first practical backend target because the intended Windows PC has:

- Intel i7-14700K
- 64GB RAM
- NVIDIA RTX 4080 SUPER with 16GB VRAM

That GPU is suitable for SDXL image-to-image or inpainting workflows at around 1024px using CUDA/fp16, depending on model and workflow choices.

## Deprecated Legacy Endpoint

```text
POST /preview/composite
```

This route is still present for legacy/internal behavior and is marked deprecated in FastAPI. It supports the older overlay and experimental `fit_harness` compositing pipeline.

Important limitation: overlay and `fit_harness` are not true virtual try-on. They can still look like a product image pasted over the dog photo. The generation endpoint should be used for realistic dog clothes and harness previews.

## Backend Architecture

```text
app/
  config.py
  main.py
  api/
    generate.py
    preview.py
  services/
    generation_prompt_service.py
    image_generation_service.py
    model_backend_service.py
    comfyui_client_service.py
    mask_service.py
    background_remove_service.py
    composite_service.py
  prompts/
    dog_outfit_generation_prompt.txt
    dog_outfit_negative_prompt.txt
    pet_product_preview.txt
static/
  uploads/
  results/
  debug/
```

The generation backend is pluggable through `model_backend_service.py`. Currently:

- `ComfyUIGenerationBackend` sends work to a separate local ComfyUI server.
- `LocalStubGenerationBackend` intentionally does not composite and returns a clear unavailable error.

No large models are downloaded by FastAPI at startup.

## Configuration

Defaults:

```text
COMFYUI_BASE_URL=http://127.0.0.1:8188
IMAGE_BACKEND=comfyui
```

Optional workflow template:

```text
COMFYUI_WORKFLOW_PATH=E:\AI\ComfyUI\workflows\dog_outfit_api_workflow.json
```

The ComfyUI client uploads the dog image, product image, and optional rough mask through the ComfyUI API, then submits the configured API workflow JSON. Workflow templates may use these placeholders:

```text
{{DOG_IMAGE}}
{{PRODUCT_IMAGE}}
{{MASK_IMAGE}}
{{PRODUCT_TYPE}}
{{GENERATION_PROMPT}}
{{NEGATIVE_PROMPT}}
```

## Recommended Local Generation Stack

- ComfyUI
- SDXL Inpainting or SDXL image-to-image
- IP-Adapter for product image reference
- ControlNet Canny or Depth for preserving dog pose/structure
- Optional segmentation/body mask later
- Target resolution: 1024px for SDXL
- fp16 / CUDA on the RTX 4080 SUPER 16GB VRAM

Suggested Windows folders if C drive storage is tight:

```text
E:\AI\ComfyUI
E:\AI\Models
E:\AI\outputs
```

Start ComfyUI:

```bash
python main.py --listen 127.0.0.1 --port 8188
```

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

## Install FastAPI Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The FastAPI side stays lightweight and does not include `torch` or `diffusers`. ComfyUI should be installed and managed separately.

## Generate Endpoint

`POST /preview/generate`

Multipart form fields:

- `dog_image`: uploaded dog photo.
- `product_image`: uploaded product image.
- `product_type`: required string, `harness` or `clothes`.
- `prompt_mode`: optional string, default `wearing_preview`.
- `use_comfyui`: optional boolean, default `true`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/preview/generate \
  -F "dog_image=@/path/to/dog.jpg" \
  -F "product_image=@/path/to/harness.png" \
  -F "product_type=harness"
```

Success response:

```json
{
  "status": "success",
  "result_image_path": "/static/results/generated_pet_preview_abc123.png",
  "product_type": "harness",
  "generation_prompt": "Use the dog photo as the identity and pose reference...",
  "negative_prompt": "pasted sticker, flat overlay, floating product...",
  "backend": "comfyui",
  "message": "Generated a local image-editing preview with the configured backend."
}
```

Invalid product type:

```json
{
  "status": "failed",
  "stage": "validation",
  "message": "product_type must be harness or clothes."
}
```

If ComfyUI or a generation backend is unavailable:

```json
{
  "status": "failed",
  "stage": "model_backend",
  "message": "No local image generation backend is configured or ComfyUI is not running. Start ComfyUI and configure COMFYUI_BASE_URL."
}
```

The unavailable backend response uses HTTP `501`. The endpoint does not fall back to overlay, because compositing is not image generation.

## Generation Prompt

The fixed prompt is stored at:

```text
app/prompts/dog_outfit_generation_prompt.txt
```

The fixed negative prompt is stored at:

```text
app/prompts/dog_outfit_negative_prompt.txt
```

They instruct the model to preserve the dog identity, pose, fur color, product design, color, material, pattern, shadows, folds, occlusion, and contact with fur while making the product look worn rather than pasted on.

## Limitations

- Result quality depends on the ComfyUI workflow and model selection.
- Product identity may not be perfectly preserved.
- Dog pose and product angle affect quality.
- This is not a trained pet-specific virtual try-on model yet.
- The rough torso mask is only a placeholder for future inpainting workflows.
- A proper workflow still needs suitable SDXL/IP-Adapter/ControlNet nodes and local model files in ComfyUI.
