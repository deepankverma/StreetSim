#!/usr/bin/env python3
"""CLI entry point for street image -> street_data.json (2-pass VLM flow)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from street_vlm.pipeline import save_street_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract street scene parameters from an image using an Ollama VLM and write street_data.json. "
            "This version uses a 2-pass flow: coarse extraction + refinement."
        )
    )
    parser.add_argument("--image", required=True, help="Input street-view image path.")
    parser.add_argument("--out", required=True, help="Output JSON path for street_data.json.")
    parser.add_argument("--model", default="qwen2.5vl", help="Ollama model name, e.g. qwen2.5vl:7b.")
    parser.add_argument("--policy", help="Optional JSON file overriding default mapping policy.")
    parser.add_argument("--save-vision", help="Optional path to save final refined vision JSON.")
    parser.add_argument(
        "--save-vision-passes-dir",
        help="Optional folder to save pass_1_coarse.json and pass_2_refine.json.",
    )
    parser.add_argument(
        "--print-vision-summary",
        action="store_true",
        help="Print a concise summary of what the model extracted.",
    )
    parser.add_argument(
        "--print-vision-json",
        action="store_true",
        help="Print the final refined vision JSON before mapping.",
    )
    parser.add_argument(
        "--print-vision-passes",
        action="store_true",
        help="Print raw JSON for pass 1 and pass 2 in the terminal.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for Ollama.")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/chat", help="Ollama chat API URL.")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds for each Ollama call.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip Pydantic validation of both vision JSON and final street_data.json.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    try:
        cfg = save_street_data(
            image_path=str(image_path),
            out_json=args.out,
            model=args.model,
            policy=args.policy,
            temperature=args.temperature,
            ollama_url=args.ollama_url,
            request_timeout=args.request_timeout,
            validate_schema=not args.no_validate,
            save_vision_json=args.save_vision,
            save_vision_passes_dir=args.save_vision_passes_dir,
            print_vision_summary=args.print_vision_summary,
            print_vision_json=args.print_vision_json,
            print_vision_passes=args.print_vision_passes,
            verbose=args.verbose,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(cfg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
