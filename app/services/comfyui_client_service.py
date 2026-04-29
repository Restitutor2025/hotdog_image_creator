"""ComfyUI API client for local GPU image generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests


class ComfyUIClientError(RuntimeError):
    """Base error for ComfyUI client failures."""


class ComfyUIUnavailableError(ComfyUIClientError):
    """Raised when ComfyUI cannot be reached or is not configured."""


class ComfyUIWorkflowError(ComfyUIClientError):
    """Raised when workflow submission or completion fails."""


class ComfyUIClient:
    """Thin client around ComfyUI's HTTP API."""

    def __init__(
        self,
        base_url: str,
        workflow_path: str | None,
        timeout_seconds: float,
        poll_interval_seconds: float,
        prompt_timeout_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.workflow_path = Path(workflow_path) if workflow_path else None
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.prompt_timeout_seconds = prompt_timeout_seconds
        self.client_id = uuid4().hex

    def check_health(self) -> None:
        """Verify that ComfyUI is reachable."""

        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ComfyUIUnavailableError("ComfyUI is not reachable.") from exc

    def upload_image(self, image_path: Path) -> dict[str, str]:
        """Upload an image to the ComfyUI input folder."""

        try:
            with image_path.open("rb") as image_file:
                response = requests.post(
                    f"{self.base_url}/upload/image",
                    files={"image": (image_path.name, image_file, "image/png")},
                    data={"type": "input", "overwrite": "false"},
                    timeout=self.timeout_seconds,
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ComfyUIUnavailableError("Could not upload image to ComfyUI.") from exc

        data = response.json()
        return {
            "name": data.get("name", image_path.name),
            "subfolder": data.get("subfolder", ""),
            "type": data.get("type", "input"),
        }

    def load_workflow_template(self) -> dict[str, Any]:
        """Load the configured ComfyUI API workflow JSON template."""

        if self.workflow_path is None:
            raise ComfyUIUnavailableError("COMFYUI_WORKFLOW_PATH is not configured.")
        if not self.workflow_path.exists():
            raise ComfyUIUnavailableError(f"ComfyUI workflow file does not exist: {self.workflow_path}")

        try:
            return json.loads(self.workflow_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ComfyUIWorkflowError("ComfyUI workflow file must be valid JSON.") from exc

    def build_workflow(self, replacements: dict[str, str]) -> dict[str, Any]:
        """Replace {{PLACEHOLDER}} strings anywhere inside the workflow template."""

        workflow = self.load_workflow_template()

        def replace_value(value: Any) -> Any:
            if isinstance(value, str):
                replaced = value
                for key, replacement in replacements.items():
                    replaced = replaced.replace(f"{{{{{key}}}}}", replacement)
                return replaced
            if isinstance(value, list):
                return [replace_value(item) for item in value]
            if isinstance(value, dict):
                return {key: replace_value(item) for key, item in value.items()}
            return value

        return replace_value(workflow)

    def submit_workflow(self, workflow: dict[str, Any]) -> str:
        """Submit the workflow and return the ComfyUI prompt id."""

        try:
            response = requests.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow, "client_id": self.client_id},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ComfyUIWorkflowError("Could not submit workflow to ComfyUI.") from exc

        prompt_id = response.json().get("prompt_id")
        if not prompt_id:
            raise ComfyUIWorkflowError("ComfyUI did not return a prompt_id.")
        return str(prompt_id)

    def wait_for_completion(self, prompt_id: str) -> dict[str, Any]:
        """Poll ComfyUI history until the prompt has outputs or times out."""

        deadline = time.monotonic() + self.prompt_timeout_seconds
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=self.timeout_seconds)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise ComfyUIWorkflowError("Could not read ComfyUI prompt history.") from exc

            history = response.json()
            prompt_history = history.get(prompt_id)
            if prompt_history:
                status = prompt_history.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyUIWorkflowError("ComfyUI workflow failed.")
                outputs = prompt_history.get("outputs", {})
                if outputs:
                    return prompt_history

            time.sleep(self.poll_interval_seconds)

        raise ComfyUIWorkflowError("ComfyUI prompt timed out before producing an output image.")

    def download_first_output_image(self, prompt_history: dict[str, Any], result_dir: Path) -> Path:
        """Download the first image output from ComfyUI into static/results."""

        for output in prompt_history.get("outputs", {}).values():
            for image_info in output.get("images", []):
                params = {
                    "filename": image_info.get("filename", ""),
                    "subfolder": image_info.get("subfolder", ""),
                    "type": image_info.get("type", "output"),
                }
                try:
                    response = requests.get(f"{self.base_url}/view", params=params, timeout=self.timeout_seconds)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    raise ComfyUIWorkflowError("Could not download ComfyUI output image.") from exc

                result_dir.mkdir(parents=True, exist_ok=True)
                suffix = Path(params["filename"]).suffix or ".png"
                result_path = result_dir / f"generated_pet_preview_{uuid4().hex}{suffix}"
                result_path.write_bytes(response.content)
                return result_path

        raise ComfyUIWorkflowError("ComfyUI completed without an image output.")

    def generate_image(
        self,
        dog_image_path: Path,
        product_image_path: Path,
        mask_image_path: Path | None,
        product_type: str,
        generation_prompt: str,
        negative_prompt: str,
        result_dir: Path,
    ) -> Path:
        """Upload inputs, submit the configured workflow, and save the generated image."""

        self.check_health()
        dog_upload = self.upload_image(dog_image_path)
        product_upload = self.upload_image(product_image_path)
        mask_upload = self.upload_image(mask_image_path) if mask_image_path else {"name": "", "subfolder": "", "type": ""}

        workflow = self.build_workflow(
            {
                "DOG_IMAGE": dog_upload["name"],
                "DOG_IMAGE_SUBFOLDER": dog_upload["subfolder"],
                "DOG_IMAGE_TYPE": dog_upload["type"],
                "PRODUCT_IMAGE": product_upload["name"],
                "PRODUCT_IMAGE_SUBFOLDER": product_upload["subfolder"],
                "PRODUCT_IMAGE_TYPE": product_upload["type"],
                "MASK_IMAGE": mask_upload["name"],
                "MASK_IMAGE_SUBFOLDER": mask_upload["subfolder"],
                "MASK_IMAGE_TYPE": mask_upload["type"],
                "PRODUCT_TYPE": product_type,
                "GENERATION_PROMPT": generation_prompt,
                "NEGATIVE_PROMPT": negative_prompt,
            }
        )
        prompt_id = self.submit_workflow(workflow)
        prompt_history = self.wait_for_completion(prompt_id)
        return self.download_first_output_image(prompt_history, result_dir)
