#!/usr/bin/env python3
"""Run the SOLWEIG raster export from an existing .blend file.

This wraps the Blender-side exporter and optionally converts the ASCII grids to
GeoTIFF using rasterio in the host Python environment.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent


def quote_cmd(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_cmd(cmd: list[object], dry_run: bool = False) -> None:
    print(f"RUN: {quote_cmd(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export SOLWEIG-style rasters from a Blender file.")
    ap.add_argument("--blend", required=True, help="Input .blend file")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--blender", default="blender", help="Blender executable path")
    ap.add_argument(
        "--scripts-dir",
        help="Folder containing solweig_export_rasters.py. Defaults to this script's folder.",
    )
    ap.add_argument("--cellsize", type=float, default=1.0)
    ap.add_argument("--padding", type=float, default=2.0)
    ap.add_argument("--origin-x", type=float)
    ap.add_argument("--origin-y", type=float)
    ap.add_argument("--epsg", type=int)
    ap.add_argument("--tree-class", default="deciduous", choices=["deciduous", "evergreen"])
    ap.add_argument("--ground-class", default="bare_soil", choices=["bare_soil", "grass", "paved"])
    ap.add_argument("--include-vehicles", action="store_true")
    ap.add_argument("--include-humans", action="store_true")
    ap.add_argument("--include-lamps", action="store_true")
    ap.add_argument("--keep-ground-buffer", action="store_true")
    ap.add_argument("--write-npy", action="store_true")
    ap.add_argument("--skip-geotiff", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def find_script(scripts_dir: Path, name: str) -> Path:
    for candidate in (
        scripts_dir / name,
        scripts_dir / "solweig" / name,
        THIS_DIR / name,
    ):
        if candidate.exists():
            return candidate
    return scripts_dir / name


def main() -> int:
    args = parse_args()
    blend = Path(args.blend).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    scripts_dir = Path(args.scripts_dir).expanduser().resolve() if args.scripts_dir else THIS_DIR
    exporter = find_script(scripts_dir, "solweig_export_rasters.py")
    converter = find_script(scripts_dir, "solweig_ascii_to_geotiff.py")

    if not blend.exists():
        print(f"Error: missing blend: {blend}", file=sys.stderr)
        return 1
    if not exporter.exists():
        print(f"Error: missing exporter: {exporter}", file=sys.stderr)
        return 1

    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.blender,
        "-b",
        blend,
        "-P",
        exporter,
        "--",
        "--outdir",
        outdir,
        "--cellsize",
        args.cellsize,
        "--padding",
        args.padding,
        "--tree-class",
        args.tree_class,
        "--ground-class",
        args.ground_class,
        "--include-vehicles",
        str(bool(args.include_vehicles)).lower(),
        "--include-humans",
        str(bool(args.include_humans)).lower(),
        "--include-lamps",
        str(bool(args.include_lamps)).lower(),
        "--keep-ground-buffer",
        str(bool(args.keep_ground_buffer)).lower(),
        "--write-npy",
        str(bool(args.write_npy)).lower(),
        "--verbose",
        "true",
    ]
    if args.origin_x is not None:
        cmd.extend(["--origin-x", args.origin_x])
    if args.origin_y is not None:
        cmd.extend(["--origin-y", args.origin_y])
    if args.epsg is not None:
        cmd.extend(["--epsg", args.epsg])

    run_cmd(cmd, dry_run=args.dry_run)

    if not args.skip_geotiff and converter.exists():
        cmd2 = [
            sys.executable,
            converter,
            "--indir",
            outdir,
        ]
        if args.epsg is not None:
            cmd2.extend(["--epsg", args.epsg])
        run_cmd(cmd2, dry_run=args.dry_run)

    print("\nSOLWEIG raster export completed.")
    print(f"ASCII rasters: {outdir}")
    if not args.skip_geotiff:
        print(f"GeoTIFF rasters: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
