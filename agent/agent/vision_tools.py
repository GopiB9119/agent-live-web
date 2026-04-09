import base64
import json
from pathlib import Path
try:
    from runtime_utils import redact_sensitive_data as _redact_sensitive_data
except Exception:
    from .runtime_utils import redact_sensitive_data as _redact_sensitive_data


class VisionManager:
    """
    Vision tools: capture screenshots, encode images for LLM analysis,
    compare visual states, and extract structured data from screenshots.
    """

    def __init__(self, workspace_root: Path, resolve_workspace_path_fn):
        self.workspace_root = Path(workspace_root).resolve()
        self.resolve_workspace_path = resolve_workspace_path_fn

    @staticmethod
    def _json_response(payload, max_chars=30000):
        return json.dumps(_redact_sensitive_data(payload, max_chars=max_chars), ensure_ascii=True)

    def _encode_image_to_base64(self, image_path: Path) -> str:
        """Read an image and return base64-encoded string."""
        data = image_path.read_bytes()
        return base64.b64encode(data).decode("ascii")

    def _image_mime_type(self, path_obj: Path) -> str:
        suffix = path_obj.suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(suffix, "image/png")

    async def vision_encode(self, kwargs_dict):
        """Encode a local image to base64 for LLM vision input."""
        kwargs = kwargs_dict or {}
        path_value = str(kwargs.get("path", "")).strip()
        max_bytes = int(kwargs.get("max_bytes", 4 * 1024 * 1024))

        if not path_value:
            return self._json_response({"status": "failed", "error": "path is required"})

        try:
            image_path = self.resolve_workspace_path(path_value, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        if not image_path.is_file():
            return self._json_response({"status": "failed", "error": f"Not a file: {image_path}"})

        size = image_path.stat().st_size
        if size > max_bytes:
            return self._json_response({
                "status": "failed",
                "error": f"Image too large ({size} bytes, max {max_bytes})",
                "path": path_value,
            })

        mime = self._image_mime_type(image_path)
        b64 = self._encode_image_to_base64(image_path)

        return self._json_response({
            "status": "ok",
            "path": image_path.relative_to(self.workspace_root).as_posix(),
            "mime_type": mime,
            "size_bytes": size,
            "base64_length": len(b64),
            "data_url": f"data:{mime};base64,{b64[:100]}...",
            "base64": b64,
        }, max_chars=max_bytes * 2)

    async def vision_compare(self, kwargs_dict):
        """Compare two screenshots and report structural differences."""
        kwargs = kwargs_dict or {}
        before_path = str(kwargs.get("before", "")).strip()
        after_path = str(kwargs.get("after", "")).strip()

        if not before_path or not after_path:
            return self._json_response({"status": "failed", "error": "before and after paths are required"})

        try:
            before = self.resolve_workspace_path(before_path, must_exist=True)
            after = self.resolve_workspace_path(after_path, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        before_size = before.stat().st_size
        after_size = after.stat().st_size
        size_diff = after_size - before_size
        size_changed = abs(size_diff) > 1024

        # Byte-level comparison
        before_bytes = before.read_bytes()
        after_bytes = after.read_bytes()
        identical = before_bytes == after_bytes

        return self._json_response({
            "status": "ok",
            "identical": identical,
            "before": {"path": before_path, "size": before_size},
            "after": {"path": after_path, "size": after_size},
            "size_diff_bytes": size_diff,
            "visually_changed": not identical,
            "significant_size_change": size_changed,
            "before_base64": self._encode_image_to_base64(before) if not identical else None,
            "after_base64": self._encode_image_to_base64(after) if not identical else None,
        }, max_chars=8 * 1024 * 1024)

    async def vision_describe_page(self, kwargs_dict):
        """Build a structured vision analysis request for an LLM with vision capability."""
        kwargs = kwargs_dict or {}
        screenshot_path = str(kwargs.get("screenshot_path", "")).strip()
        question = str(kwargs.get("question", "Describe what you see on this page.")).strip()
        page_url = str(kwargs.get("page_url", "")).strip()
        page_title = str(kwargs.get("page_title", "")).strip()

        if not screenshot_path:
            return self._json_response({"status": "failed", "error": "screenshot_path is required"})

        try:
            image_path = self.resolve_workspace_path(screenshot_path, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        mime = self._image_mime_type(image_path)
        b64 = self._encode_image_to_base64(image_path)

        # Build a structured message for vision-capable LLM
        context_parts = []
        if page_url:
            context_parts.append(f"Page URL: {page_url}")
        if page_title:
            context_parts.append(f"Page title: {page_title}")
        context = "\n".join(context_parts)

        vision_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{context}\n\n{question}" if context else question,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                    },
                },
            ],
        }

        return self._json_response({
            "status": "ok",
            "screenshot_path": image_path.relative_to(self.workspace_root).as_posix(),
            "question": question,
            "image_size_bytes": image_path.stat().st_size,
            "vision_message": vision_message,
        }, max_chars=8 * 1024 * 1024)
