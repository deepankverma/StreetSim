"""Ollama client for image + prompt JSON extraction.

This version uses a 2-pass flow:
1) coarse extraction
2) refinement
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"


class OllamaResponseError(RuntimeError):
    """Raised when the model does not return valid JSON."""


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def image_to_base64(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _extract_json_text(content: str) -> str:
    content = content.strip()
    if not content:
        raise OllamaResponseError("Empty response from Ollama.")

    fenced = _JSON_BLOCK_RE.search(content)
    if fenced:
        return fenced.group(1).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return content


def ask_ollama_json(
    model: str,
    prompt: str,
    image_b64: str,
    temperature: float = 0.0,
    ollama_url: str = OLLAMA_URL,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
    }
    try:
        response = requests.post(
            ollama_url,
            json=payload,
            timeout=timeout,
            headers={
                "ngrok-skip-browser-warning": "true",
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            f"Timed out waiting for Ollama after {timeout:.0f}s for a single request. "
            f"Try --request-timeout 900 or 1200, use a smaller model, or keep the new 2-pass flow."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request to Ollama failed: {exc}") from exc

    data = response.json()
    try:
        content = data["message"]["content"]
    except KeyError as exc:
        raise OllamaResponseError(f"Unexpected Ollama response shape: {data}") from exc

    json_text = _extract_json_text(content)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise OllamaResponseError(f"Model returned invalid JSON:\n{content}") from exc


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_vision_schema(
    image_path: str,
    model: str,
    prompt1: str,
    prompt2: str,
    temperature: float = 0.0,
    ollama_url: str = OLLAMA_URL,
    request_timeout: float = 600.0,
    save_passes_dir: Optional[str] = None,
    print_passes: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    if verbose:
        print("[vision] Encoding image...", flush=True)
    image_b64 = image_to_base64(image_path)

    save_dir = Path(save_passes_dir) if save_passes_dir else None

    if verbose:
        print("[vision] Pass 1/2: coarse extraction...", flush=True)
    coarse = ask_ollama_json(
        model,
        prompt1,
        image_b64,
        temperature=temperature,
        ollama_url=ollama_url,
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
    refine = ask_ollama_json(
        model,
        prompt2 + "\n\nDRAFT_JSON:\n" + json.dumps(coarse, ensure_ascii=False),
        image_b64,
        temperature=temperature,
        ollama_url=ollama_url,
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
