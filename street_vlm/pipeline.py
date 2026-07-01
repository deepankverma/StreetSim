"""End-to-end pipeline from image to street_data.json.

2-pass VLM flow:
1) coarse extraction
2) refinement

This version adds robust enum coercion before schema validation so minor model
formatting issues (spaces, pipe-separated menus, weighted menu echoes, etc.) do
not break the run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .defaults import DEFAULT_POLICY
from .mapper import normalize_street_data, vision_to_street_data
from .ollama_client import extract_vision_schema as extract_ollama_vision_schema
from .openai_client import OPENAI_RESPONSES_URL, extract_vision_schema as extract_openai_vision_schema
from .prompts import COARSE_PROMPT, REFINE_PROMPT
from .schemas import StreetDataConfig, VisionStreetSchema


def _load_policy(policy: Optional[Dict[str, Any] | str]) -> Dict[str, Any]:
    if policy is None:
        return dict(DEFAULT_POLICY)
    if isinstance(policy, dict):
        merged = dict(DEFAULT_POLICY)
        merged.update(policy)
        return merged
    path = Path(policy)
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_POLICY)
    merged.update(data)
    return merged


def _validate_and_dump(model_cls: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data).model_dump()
    return model_cls.parse_obj(data).dict()


def _norm_text(value: Any) -> str:
    s = str(value).strip().lower()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _extract_number(token: str) -> float:
    nums = re.findall(r"(?<![a-z])(\d+(?:\.\d+)?)", token)
    if not nums:
        return 0.0
    try:
        return float(nums[-1])
    except Exception:
        return 0.0


def _coerce_enum(value: Any, canonical_patterns: Dict[str, tuple[str, ...]], default: str) -> str:
    """Coerce messy model output to one of a small enum set.

    Handles cases like:
    - 'street level' -> 'street_level'
    - 'urban_ street' -> 'urban_street'
    - 'urban_0.5|suburban_0.5|residential_0.5|unknown' -> 'urban_street'
    - 'street_level|elevated|unknown' -> 'street_level'
    """
    if value is None:
        return default

    raw = _norm_text(value)
    if not raw:
        return default

    # Exact alias/direct match first.
    for canonical, pats in canonical_patterns.items():
        if raw == canonical or raw in pats:
            return canonical

    # Split menus/choice echoes and score candidates.
    pieces = [p for p in re.split(r"[|,;/\n]+", raw) if p.strip()]
    best_choice = None
    best_score = float("-inf")

    for idx, piece in enumerate(pieces):
        piece_norm = _norm_text(piece)
        compact = piece_norm.replace("_", "")
        for canonical, pats in canonical_patterns.items():
            for pat in pats:
                pat_norm = _norm_text(pat)
                pat_compact = pat_norm.replace("_", "")
                # allow exact, startswith, or contained compact match
                matched = (
                    piece_norm == pat_norm
                    or compact == pat_compact
                    or piece_norm.startswith(pat_norm + "_")
                    or piece_norm.startswith(pat_compact + "_")
                    or pat_compact in compact
                )
                if matched:
                    score = _extract_number(piece_norm)
                    # Prefer earlier item on ties because models often echo options in order.
                    score = score - idx * 1e-6
                    if score > best_score:
                        best_choice = canonical
                        best_score = score
                    break

    if best_choice is not None:
        return best_choice

    # Fallback: search the whole string for canonical patterns.
    compact_raw = raw.replace("_", "")
    for canonical, pats in canonical_patterns.items():
        for pat in pats:
            pat_compact = _norm_text(pat).replace("_", "")
            if pat_compact in compact_raw:
                return canonical

    return default


def _normalize_vision_enums(vision: Dict[str, Any]) -> Dict[str, Any]:
    scene_patterns = {
        "urban_street": ("urban_street", "urbanstreet", "urban"),
        "suburban_street": ("suburban_street", "suburbanstreet", "suburban"),
        "residential_street": ("residential_street", "residentialstreet", "residential"),
        "unknown": ("unknown",),
    }
    camera_patterns = {
        "street_level": ("street_level", "streetlevel", "street_view", "streetview", "ground_level", "groundlevel"),
        "elevated": ("elevated", "elevated_view", "birdseye", "birds_eye"),
        "unknown": ("unknown",),
    }

    vision["scene_type"] = _coerce_enum(vision.get("scene_type"), scene_patterns, "unknown")
    vision["camera_view"] = _coerce_enum(vision.get("camera_view"), camera_patterns, "unknown")
    return vision


def _fmt_bool(v: bool) -> str:
    return "yes" if v else "no"


def _print_vision_summary(vision: Dict[str, Any]) -> None:
    def width_block(side: Dict[str, Any], key: str) -> str:
        node = side.get(key, {})
        if key == "parking":
            return (
                f"present={_fmt_bool(bool(node.get('present', False)))}, "
                f"bay_width_m={node.get('bay_width_m', 0)}, "
                f"count_visible={node.get('count_visible', 0)}, "
                f"conf={node.get('confidence', 0)}"
            )
        if key in ("buildings", "trees"):
            return (
                f"present={_fmt_bool(bool(node.get('present', False)))}, "
                f"count_visible={node.get('count_visible', 0)}, "
                f"conf={node.get('confidence', 0)}"
            )
        return (
            f"present={_fmt_bool(bool(node.get('present', False)))}, "
            f"width_m={node.get('width_m', 0)}, "
            f"conf={node.get('confidence', 0)}"
        )

    print("\n=== Vision summary ===", flush=True)
    print(f"scene_type: {vision.get('scene_type', 'unknown')}", flush=True)
    print(f"camera_view: {vision.get('camera_view', 'unknown')}", flush=True)
    for side_name in ("left_side", "right_side"):
        side = vision.get(side_name, {})
        print(f"\n[{side_name}]", flush=True)
        for key in ("driveway", "bikepath", "footpath", "buildings", "trees", "parking"):
            print(f"  - {key}: {width_block(side, key)}", flush=True)
    median = vision.get("median", {})
    print("\n[median]", flush=True)
    print(
        "  - present={present}, width_m={width}, trees_present={trees_present}, tree_count_visible={count}, conf={conf}".format(
            present=_fmt_bool(bool(median.get("present", False))),
            width=median.get("width_m", 0),
            trees_present=_fmt_bool(bool(median.get("trees_present", False))),
            count=median.get("tree_count_visible", 0),
            conf=median.get("confidence", 0),
        ),
        flush=True,
    )
    dynamic = vision.get("dynamic", {})
    print("\n[dynamic]", flush=True)
    print(f"  - parked_cars_visible: {dynamic.get('parked_cars_visible', 0)}", flush=True)
    print(f"  - moving_cars_visible: {dynamic.get('moving_cars_visible', 0)}", flush=True)
    print(f"  - pedestrians_visible: {dynamic.get('pedestrians_visible', 0)}", flush=True)
    vegetation = vision.get("vegetation", {})
    print("\n[vegetation]", flush=True)
    print(
        f"  - present={_fmt_bool(bool(vegetation.get('present', False)))}, "
        f"density_score_0_100={vegetation.get('density_score_0_100', 0)}, "
        f"conf={vegetation.get('confidence', 0)}",
        flush=True,
    )
    if vision.get("street_length_m_estimate") is not None:
        print(f"\nstreet_length_m_estimate: {vision.get('street_length_m_estimate')}", flush=True)
    if vision.get("global_notes"):
        print(f"global_notes: {vision.get('global_notes')}", flush=True)
    if vision.get("unresolvable"):
        print(f"unresolvable: {vision.get('unresolvable')}", flush=True)
    print("=== End vision summary ===\n", flush=True)


def _extract_vision_with_provider(
    provider: str,
    image_path: str,
    model: str,
    prompt1: str,
    prompt2: str,
    temperature: float,
    ollama_url: str,
    openai_api_key: Optional[str],
    openai_base_url: str,
    request_timeout: float,
    save_passes_dir: Optional[str] = None,
    print_passes: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    provider_key = (provider or "ollama").strip().lower()
    if provider_key == "ollama":
        return extract_ollama_vision_schema(
            image_path=image_path,
            model=model,
            prompt1=prompt1,
            prompt2=prompt2,
            temperature=temperature,
            ollama_url=ollama_url,
            request_timeout=request_timeout,
            save_passes_dir=save_passes_dir,
            print_passes=print_passes,
            verbose=verbose,
        )
    if provider_key in {"openai", "responses"}:
        return extract_openai_vision_schema(
            image_path=image_path,
            model=model,
            prompt1=prompt1,
            prompt2=prompt2,
            temperature=temperature,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            request_timeout=request_timeout,
            save_passes_dir=save_passes_dir,
            print_passes=print_passes,
            verbose=verbose,
        )
    raise ValueError("Unknown VLM provider '{0}'. Use 'ollama' or 'openai'.".format(provider))


def image_to_street_data(
    image_path: str,
    model: str = "qwen2.5vl",
    provider: str = "ollama",
    policy: Optional[Dict[str, Any] | str] = None,
    temperature: float = 0.0,
    ollama_url: str = "http://localhost:11434/api/chat",
    openai_api_key: Optional[str] = None,
    openai_base_url: str = OPENAI_RESPONSES_URL,
    request_timeout: float = 600.0,
    validate_schema: bool = True,
    print_vision_summary: bool = False,
    print_vision_json: bool = False,
    print_vision_passes: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    active_policy = _load_policy(policy)
    if verbose:
        print(f"[1/4] Running 2-pass vision extraction with provider '{provider}'...", flush=True)
    vision = _extract_vision_with_provider(
        provider=provider,
        image_path=image_path,
        model=model,
        prompt1=COARSE_PROMPT,
        prompt2=REFINE_PROMPT,
        temperature=temperature,
        ollama_url=ollama_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        request_timeout=request_timeout,
        print_passes=print_vision_passes,
        verbose=verbose,
    )
    vision = _normalize_vision_enums(vision)
    if validate_schema:
        if verbose:
            print("[2/4] Validating vision schema...", flush=True)
        vision = _validate_and_dump(VisionStreetSchema, vision)
    if print_vision_summary:
        _print_vision_summary(vision)
    if print_vision_json:
        print(json.dumps(vision, indent=2, ensure_ascii=False), flush=True)

    if verbose:
        print("[3/4] Mapping to street_data.json structure...", flush=True)
    cfg = vision_to_street_data(vision, policy=active_policy)
    cfg = normalize_street_data(cfg)
    if validate_schema:
        if verbose:
            print("[4/4] Validating final street_data.json schema...", flush=True)
        cfg = _validate_and_dump(StreetDataConfig, cfg)
    return cfg


def save_street_data(
    image_path: str,
    out_json: str,
    model: str = "qwen2.5vl",
    provider: str = "ollama",
    policy: Optional[Dict[str, Any] | str] = None,
    temperature: float = 0.0,
    ollama_url: str = "http://localhost:11434/api/chat",
    openai_api_key: Optional[str] = None,
    openai_base_url: str = OPENAI_RESPONSES_URL,
    request_timeout: float = 600.0,
    validate_schema: bool = True,
    save_vision_json: Optional[str] = None,
    save_vision_passes_dir: Optional[str] = None,
    print_vision_summary: bool = False,
    print_vision_json: bool = False,
    print_vision_passes: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    if verbose:
        print("[1/6] Loading policy...", flush=True)
    active_policy = _load_policy(policy)

    if verbose:
        print(f"[2/6] Running 2-pass vision extraction with provider '{provider}' and model '{model}'...", flush=True)
    vision = _extract_vision_with_provider(
        provider=provider,
        image_path=image_path,
        model=model,
        prompt1=COARSE_PROMPT,
        prompt2=REFINE_PROMPT,
        temperature=temperature,
        ollama_url=ollama_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        request_timeout=request_timeout,
        save_passes_dir=save_vision_passes_dir,
        print_passes=print_vision_passes,
        verbose=verbose,
    )
    vision = _normalize_vision_enums(vision)
    if validate_schema:
        if verbose:
            print("[3/6] Validating vision schema...", flush=True)
        vision = _validate_and_dump(VisionStreetSchema, vision)

    if save_vision_json:
        if verbose:
            print(f"[4/6] Writing intermediate vision JSON to {save_vision_json}...", flush=True)
        Path(save_vision_json).write_text(json.dumps(vision, indent=2, ensure_ascii=False), encoding="utf-8")

    if print_vision_summary:
        _print_vision_summary(vision)
    if print_vision_json:
        print(json.dumps(vision, indent=2, ensure_ascii=False), flush=True)

    if verbose:
        print("[5/6] Mapping vision output to street_data.json structure...", flush=True)
    cfg = vision_to_street_data(vision, policy=active_policy)
    cfg = normalize_street_data(cfg)
    if validate_schema:
        if verbose:
            print("[5/6] Validating final street_data.json schema...", flush=True)
        cfg = _validate_and_dump(StreetDataConfig, cfg)

    if verbose:
        print(f"[6/6] Writing final output to {out_json}...", flush=True)
    Path(out_json).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print("[done] Finished successfully.", flush=True)
    return cfg
