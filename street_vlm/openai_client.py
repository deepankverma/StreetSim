"""OpenAI Responses API client for image + prompt JSON extraction.

This mirrors the Ollama two-pass flow while keeping the pipeline dependency
lightweight: it uses requests directly instead of requiring the OpenAI SDK.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIResponseError(RuntimeError):
    """Raised when the provider does not return valid JSON."""


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def image_to_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _extract_json_text(content: str) -> str:
    content = content.strip()
    if not content:
        raise OpenAIResponseError("Empty response from OpenAI-compatible provider.")

    fenced = _JSON_BLOCK_RE.search(content)
    if fenced:
        return fenced.group(1).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return content


def _extract_output_text(data: Dict[str, Any]) -> str:
    texts: list[str] = []

    output_text = data.get("output_text")
    if isinstance(output_text, str):
        texts.append(output_text)

    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)

    text = "\n".join(t for t in texts if t).strip()
    if not text:
        raise OpenAIResponseError(f"Unexpected OpenAI-compatible response shape: {data}")
    return text


def ask_openai_json(
    model: str,
    prompt: str,
    image_data_url: str,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    base_url: str = OPENAI_RESPONSES_URL,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    token = api_key or os.environ.get("OPENAI_API_KEY")
    if not token:
        raise OpenAIResponseError("OPENAI_API_KEY is not set. Set it or pass --openai-api-key.")

    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    }
    if temperature is not None:
        payload["temperature"] = temperature

    try:
        response = requests.post(
            base_url,
            json=payload,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            f"Timed out waiting for OpenAI-compatible provider after {timeout:.0f}s for a single request. "
            "Try --request-timeout 900 or 1200, or use a faster vision model."
        ) from exc
    except requests.exceptions.RequestException as exc:
        body = ""
        if getattr(exc, "response", None) is not None:
            body = f"\nResponse body: {exc.response.text}"
        raise RuntimeError(f"Request to OpenAI-compatible provider failed: {exc}{body}") from exc

    content = _extract_output_text(response.json())
    json_text = _extract_json_text(content)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise OpenAIResponseError(f"Provider returned invalid JSON:\n{content}") from exc


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_vision_schema(
    image_path: str,
    model: str,
    prompt1: str,
    prompt2: str,
    temperature: float = 0.0,
    openai_api_key: Optional[str] = None,
    openai_base_url: str = OPENAI_RESPONSES_URL,
    request_timeout: float = 600.0,
    save_passes_dir: Optional[str] = None,
    print_passes: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    if verbose:
        print("[vision] Encoding image...", flush=True)
    image_data_url = image_to_data_url(image_path)

    save_dir = Path(save_passes_dir) if save_passes_dir else None

    if verbose:
        print("[vision] Pass 1/2: coarse extraction...", flush=True)
    coarse = ask_openai_json(
        model,
        prompt1,
        image_data_url,
        temperature=temperature,
        api_key=openai_api_key,
        base_url=openai_base_url,
        timeout=request_timeout,
    )
    if print_passes:
        print("\n=== Vision pass 1/2: coarse extraction ===", flush=True)
        print(json.dumps(coarse, indent=2), flush=True)
        print("=== End vision pass 1/2 ===\n", flush=True)
    if save_dir:
        _write_json(save_dir / "pass_1_coarse.json", coarse)

    if verbose:
        print("[vision] Pass 2/2: refinement...", flush=True)
    refine = ask_openai_json(
        model,
        prompt2 + "\n\nDRAFT_JSON:\n" + json.dumps(coarse, ensure_ascii=False),
        image_data_url,
        temperature=temperature,
        api_key=openai_api_key,
        base_url=openai_base_url,
        timeout=request_timeout,
    )
    if print_passes:
        print("\n=== Vision pass 2/2: refinement ===", flush=True)
        print(json.dumps(refine, indent=2), flush=True)
        print("=== End vision pass 2/2 ===\n", flush=True)
    if save_dir:
        _write_json(save_dir / "pass_2_refine.json", refine)

    if verbose:
        print("[vision] Vision extraction complete.", flush=True)
    return refine
