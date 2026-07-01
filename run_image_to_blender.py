#!/usr/bin/env python3
"""Run the full image -> street_data.json -> Blender pipeline.

Adds an optional post-processing stage that uses the latest pipeline .blend
(preferably 04_sounded.blend) to:
- generate property visualization .blend files and rendered images for
  enclosure, isovist, SVF, and shade fraction
- generate a metrics JSON using Prop_5_6_Loud_dyna.py

This runner remains tolerant to slightly different
street_vlm.pipeline.save_street_data signatures, so it stays usable across
small patch versions.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

try:
    from street_vlm.pipeline import save_street_data
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import street_vlm.pipeline. Place this file next to cli.py and the street_vlm package, "
        "or run it from your project root.\n"
        f"Import error: {exc}"
    )


PROPERTY_SCRIPT_NAMES = {
    "enclosure": "Prop_1_enclosure.py",
    "isovist": "Prop_2_isovist.py",
    "svf": "Prop_3_svf.py",
    "shade": "Prop_4_shadefrac.py",
}
METRICS_SCRIPT_NAME = "Prop_5_6_Loud_dyna.py"
DEFAULT_PROPERTY_LIST = ["enclosure", "isovist", "svf", "shade"]
SOLWEIG_EXPORT_SCRIPT_NAME = "solweig_export_rasters.py"
SOLWEIG_GEOTIFF_SCRIPT_NAME = "solweig_ascii_to_geotiff.py"
ENVIMET_EXPORT_SCRIPT_NAME = "blend_to_envimet_voxels.py"
POINTCLOUD_EXPORT_SCRIPT_NAME = "blend_to_pointcloud.py"

SCRIPT_SUBDIRS = {
    "01_model.py": ("modelling",),
    "02_textured.py": ("modelling",),
    "03_animated.py": ("modelling",),
    "04_soundscapes.py": ("modelling",),
    "05_render.py": ("modelling",),
    "Prop_1_enclosure.py": ("properties",),
    "Prop_2_isovist.py": ("properties",),
    "Prop_3_svf.py": ("properties",),
    "Prop_4_shadefrac.py": ("properties",),
    "Prop_5_6_Loud_dyna.py": ("properties",),
    "solweig_export_rasters.py": ("solweig",),
    "solweig_ascii_to_geotiff.py": ("solweig",),
    "blend_to_envimet_voxels.py": ("envimet",),
    "blend_to_pointcloud.py": ("pointcloud",),
}


def _quote_cmd(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_cmd(cmd: list[object], cwd: Optional[Path] = None, dry_run: bool = False) -> None:
    print(f"RUN: {_quote_cmd(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, check=True)


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {label} at '{path}', but it was not created.")


def _find_script(scripts_dir: Path, name: str) -> Optional[Path]:
    search_roots = [scripts_dir]
    parent = scripts_dir.parent
    if parent not in search_roots:
        search_roots.append(parent)

    for root in search_roots:
        direct = root / name
        if direct.exists():
            return direct
        for subdir in SCRIPT_SUBDIRS.get(name, ()):
            candidate = root / subdir / name
            if candidate.exists():
                return candidate

    for root in search_roots:
        try:
            matches = sorted(
                p for p in root.rglob(name)
                if p.is_file() and "__pycache__" not in p.parts
            )
        except OSError:
            matches = []
        if matches:
            return matches[0]
    return None


def _script_path(scripts_dir: Path, name: str) -> Path:
    path = _find_script(scripts_dir, name)
    if path is None:
        raise FileNotFoundError(f"Could not find script '{name}' under '{scripts_dir}' or its standard subfolders.")
    return path



def _render_output_name(render_mode: str, render_outname: Optional[str]) -> str:
    if render_outname:
        return render_outname
    return "spin.mp4" if render_mode == "render1" else "still.png"


def _render_resolution_args(args) -> list[object]:
    cmd: list[object] = []
    if args.render_resx is not None:
        cmd.extend(["--resx", args.render_resx])
    if args.render_resy is not None:
        cmd.extend(["--resy", args.render_resy])
    return cmd


def _render_motion_args(args) -> list[object]:
    cmd: list[object] = [
        "--pan-deg",
        args.render_pan_deg,
        "--pan-center-deg",
        args.render_pan_center_deg,
        "--exposure",
        args.render_exposure,
    ]
    if args.render_rotations is not None:
        cmd.extend(["--rotations", args.render_rotations])
    return cmd



def _csv_to_property_list(raw: str) -> list[str]:
    values = [v.strip().lower() for v in (raw or "").split(",") if v.strip()]
    if not values:
        return list(DEFAULT_PROPERTY_LIST)

    bad = [v for v in values if v not in PROPERTY_SCRIPT_NAMES]
    if bad:
        allowed = ", ".join(DEFAULT_PROPERTY_LIST)
        raise ValueError(f"Unknown property name(s): {', '.join(bad)}. Allowed values: {allowed}")
    return values



def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate street_data.json from an image, run Blender model/texture/animate/sound/render stages, "
            "then optionally create property visualizations, metrics outputs, SOLWEIG rasters, ENVI-met voxels, and point clouds from the latest .blend."
        )
    )
    ap.add_argument("--image", required=True, help="Input street-view image path.")
    ap.add_argument("--outdir", required=True, help="Output directory for JSON, blend files, and final render.")
    ap.add_argument("--blender", default="blender", help="Blender executable path. Default: blender")
    ap.add_argument(
        "--scripts-dir",
        default=".",
        help=(
            "Project root or scripts folder. The runner searches this folder and standard subfolders "
            "such as modelling/, properties/, solweig/, envimet/, and pointcloud/."
        ),
    )
    ap.add_argument(
        "--assets-dir",
        help="Project asset root containing cars/, people/, lamp/, textures/, mixamo_fbx/, and sounds/. Defaults to --scripts-dir.",
    )

    ap.add_argument(
        "--vlm-provider",
        "--provider",
        dest="provider",
        default="ollama",
        choices=["ollama", "openai"],
        help="Vision model provider. Default: ollama.",
    )
    ap.add_argument("--model", default="qwen2.5vl", help="Vision model name, e.g. qwen2.5vl:7b or gpt-5.5")
    ap.add_argument("--policy", help="Optional JSON policy file for street_vlm mapping")
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/chat", help="Ollama chat API URL")
    ap.add_argument("--openai-api-key", help="OpenAI API key. If omitted, OPENAI_API_KEY is used.")
    ap.add_argument(
        "--openai-base-url",
        default="https://api.openai.com/v1/responses",
        help="OpenAI-compatible Responses API URL.",
    )
    ap.add_argument("--request-timeout", type=float, default=600.0, help="Per-request VLM timeout in seconds")
    ap.add_argument("--temperature", type=float, default=0.0, help="Vision model temperature")
    ap.add_argument("--no-validate", action="store_true", help="Skip VLM/final JSON schema validation")
    ap.add_argument("--save-vision", help="Optional path to save final refined vision JSON")
    ap.add_argument(
        "--save-vision-passes-dir",
        help="Optional folder to save pass_1_coarse.json and pass_2_refine.json",
    )
    ap.add_argument("--print-vision-summary", action="store_true", help="Print concise extracted scene summary")
    ap.add_argument("--print-vision-json", action="store_true", help="Print final refined vision JSON")
    ap.add_argument("--print-vision-passes", action="store_true", help="Print raw JSON for pass 1 and pass 2 in terminal")

    ap.add_argument("--render-mode", default="render3", choices=["render1", "render2", "render3"], help="05_render.py mode")
    ap.add_argument("--render-outname", help="Optional final output filename, e.g. final.png or final.mp4")
    ap.add_argument("--render-resx", type=int, help="Render width in pixels. Defaults to 640 for render1 and 2048 for render2/render3.")
    ap.add_argument("--render-resy", type=int, help="Render height in pixels. Defaults to 480 for render1 and 1536 for render2/render3.")
    ap.add_argument("--render-pan-deg", type=float, default=25.0, help="Camera pan sweep in degrees for render1. Defaults to 25.")
    ap.add_argument("--render-pan-center-deg", type=float, default=-90.0, help="Center yaw for render1 pan. Defaults to -90, looking along the street.")
    ap.add_argument("--render-rotations", type=float, help="Optional full-spin override for render1. One rotation is 360 degrees.")
    ap.add_argument("--render-exposure", type=float, default=-1.0, help="View exposure for render1. Defaults to -1.0 for half brightness.")
    ap.add_argument("--fps", type=int, default=24, help="Animation/render FPS for 03_animated.py and 05_render.py")
    ap.add_argument("--duration", type=float, default=20.0, help="Animation duration in seconds for 03_animated.py")
    ap.add_argument("--render-duration-s", type=float, default=4.0, help="Camera pan/render duration in seconds for 05_render.py")

    ap.add_argument("--skip-texture", action="store_true", help="Skip 02_textured.py")
    ap.add_argument("--skip-animate", action="store_true", help="Skip 03_animated.py")
    ap.add_argument("--skip-sound", action="store_true", help="Skip 04_soundscapes.py")
    ap.add_argument("--skip-render", action="store_true", help="Skip 05_render.py")

    ap.add_argument(
        "--skip-property-images",
        action="store_true",
        help="Skip Prop_1..Prop_4 visualization blends and rendered images.",
    )
    ap.add_argument(
        "--properties",
        default=",".join(DEFAULT_PROPERTY_LIST),
        help="Comma-separated property image list. Default: enclosure,isovist,svf,shade",
    )
    ap.add_argument(
        "--property-render-mode",
        default="render2",
        choices=["render2", "render3"],
        help="Render mode for property images. render2 preserves property colors; render3 gives clay+outlines.",
    )
    ap.add_argument(
        "--skip-metrics-json",
        action="store_true",
        help="Skip Prop_5_6_Loud_dyna.py metrics JSON generation.",
    )
    ap.add_argument(
        "--metrics-outname",
        default="street_metrics.json",
        help="Filename for the metrics JSON written by Prop_5_6_Loud_dyna.py",
    )

    ap.add_argument("--lat", type=float, default=52.52, help="Latitude used by shade and metrics scripts")
    ap.add_argument("--lon", type=float, default=13.405, help="Longitude used by shade and metrics scripts")
    ap.add_argument("--date", default="2025-06-21", help="Date used by shade and metrics scripts (YYYY-MM-DD)")
    ap.add_argument("--tstart", type=float, default=12.0, help="Local start hour for shade and metrics scripts")
    ap.add_argument("--tend", type=float, default=13.0, help="Local end hour for shade and metrics scripts")
    ap.add_argument("--tz", type=float, default=2.0, help="UTC offset hours for shade and metrics scripts")
    ap.add_argument("--north", type=float, default=0.0, help="Clockwise degrees from +Y that represent north")

    ap.add_argument(
        "--skip-solweig-export",
        action="store_true",
        help="Skip SOLWEIG raster export from the latest pipeline .blend.",
    )
    ap.add_argument(
        "--solweig-outdir",
        help="Optional output folder for SOLWEIG rasters. Default: <outdir>/solweig_inputs",
    )
    ap.add_argument("--solweig-cellsize", type=float, default=1.0, help="SOLWEIG raster cell size in scene units")
    ap.add_argument("--solweig-padding", type=float, default=2.0, help="Padding around scene bounds for SOLWEIG rasters")
    ap.add_argument("--solweig-origin-x", type=float, help="Lower-left X origin for georeferenced SOLWEIG export")
    ap.add_argument("--solweig-origin-y", type=float, help="Lower-left Y origin for georeferenced SOLWEIG export")
    ap.add_argument("--solweig-epsg", type=int, help="EPSG code written to SOLWEIG metadata/GeoTIFFs")
    ap.add_argument(
        "--solweig-tree-class",
        default="deciduous",
        choices=["deciduous", "evergreen"],
        help="Vegetation land-cover class used for tree canopies in SOLWEIG export",
    )
    ap.add_argument(
        "--solweig-ground-class",
        default="bare_soil",
        choices=["bare_soil", "grass", "paved"],
        help="Fallback land-cover class for generic ground objects in SOLWEIG export",
    )
    ap.add_argument("--solweig-include-vehicles", action="store_true", help="Include vehicles in SOLWEIG DSM export")
    ap.add_argument("--solweig-include-humans", action="store_true", help="Include humans in SOLWEIG DSM export")
    ap.add_argument("--solweig-include-lamps", action="store_true", help="Include lamps in SOLWEIG DSM export")
    ap.add_argument(
        "--solweig-drop-ground-buffer",
        action="store_true",
        help="Exclude generic ground buffer meshes named like 'ground' from SOLWEIG export",
    )
    ap.add_argument("--solweig-write-npy", action="store_true", help="Also save NumPy arrays during SOLWEIG export")
    ap.add_argument("--solweig-skip-geotiff", action="store_true", help="Skip ASCII->GeoTIFF conversion for SOLWEIG export")

    ap.add_argument("--skip-envimet-export", action="store_true", help="Skip ENVI-met voxel export from the latest pipeline .blend")
    ap.add_argument("--envimet-outdir", help="Optional output folder for ENVI-met voxels. Default: <outdir>/envimet_voxels")
    ap.add_argument("--envimet-dx", type=float, default=2.0, help="ENVI-met voxel size in X direction")
    ap.add_argument("--envimet-dy", type=float, default=2.0, help="ENVI-met voxel size in Y direction")
    ap.add_argument("--envimet-dz", type=float, default=1.0, help="ENVI-met voxel size in Z direction")
    ap.add_argument("--envimet-padding", type=float, default=2.0, help="Padding around scene bounds for ENVI-met voxel export")
    ap.add_argument("--envimet-origin-x", type=float, help="Lower-left X origin for georeferenced ENVI-met export")
    ap.add_argument("--envimet-origin-y", type=float, help="Lower-left Y origin for georeferenced ENVI-met export")
    ap.add_argument("--envimet-epsg", type=int, help="EPSG code written to ENVI-met voxel metadata")
    ap.add_argument("--envimet-tree-class", default="deciduous", choices=["deciduous", "evergreen"], help="Tree class for ENVI-met canopy semantics")
    ap.add_argument("--envimet-ground-class", default="bare_soil", choices=["bare_soil", "grass", "paved"], help="Fallback ground class for ENVI-met surfaces")
    ap.add_argument("--envimet-include-vehicles", action="store_true", help="Include vehicles in ENVI-met voxel export")
    ap.add_argument("--envimet-include-humans", action="store_true", help="Include humans in ENVI-met voxel export")
    ap.add_argument("--envimet-include-lamps", action="store_true", help="Include lamps in ENVI-met voxel export")
    ap.add_argument("--envimet-drop-ground-buffer", action="store_true", help="Exclude generic ground buffer meshes named like 'ground' from ENVI-met export")
    ap.add_argument("--envimet-canopy-min-height", type=float, default=0.5, help="Minimum canopy height threshold used in ENVI-met voxel export")
    ap.add_argument("--envimet-woody-min-height", type=float, default=0.5, help="Minimum woody height threshold used in ENVI-met voxel export")
    ap.add_argument("--envimet-skip-npz", action="store_true", help="Skip writing envimet_voxels.npz bundle")
    ap.add_argument("--envimet-skip-json-fallback", action="store_true", help="Skip writing .json.gz fallback arrays when NumPy output is unavailable")

    ap.add_argument("--skip-pointcloud-export", action="store_true", help="Skip airborne point-cloud export from the latest pipeline .blend")
    ap.add_argument("--pointcloud-outdir", help="Optional output folder for point clouds. Default: <outdir>/pointcloud")
    ap.add_argument("--pointcloud-spacing", type=float, default=0.5, help="Point spacing for airborne point-cloud export")
    ap.add_argument("--pointcloud-padding", type=float, default=2.0, help="Padding around scene bounds for point-cloud export")
    ap.add_argument("--pointcloud-origin-x", type=float, help="Lower-left X origin for georeferenced point-cloud export")
    ap.add_argument("--pointcloud-origin-y", type=float, help="Lower-left Y origin for georeferenced point-cloud export")
    ap.add_argument("--pointcloud-epsg", type=int, help="EPSG code written to point-cloud metadata")
    ap.add_argument("--pointcloud-scan-count", type=int, default=3, help="Number of top-down scan passes for point-cloud export")
    ap.add_argument("--pointcloud-jitter-frac", type=float, default=0.35, help="Fractional XY jitter applied within each point-cloud sample cell")
    ap.add_argument("--pointcloud-noise-xy", type=float, default=0.02, help="Gaussian XY noise added to point-cloud samples")
    ap.add_argument("--pointcloud-noise-z", type=float, default=0.03, help="Gaussian Z noise added to point-cloud samples")
    ap.add_argument("--pointcloud-dropout", type=float, default=0.0, help="Random dropout probability for point-cloud samples")
    ap.add_argument("--pointcloud-ground-return-prob", type=float, default=0.25, help="Probability of writing an additional ground return beneath canopy")
    ap.add_argument("--pointcloud-include-vehicles", action="store_true", help="Include vehicles in point-cloud export")
    ap.add_argument("--pointcloud-include-humans", action="store_true", help="Include humans in point-cloud export")
    ap.add_argument("--pointcloud-include-lamps", action="store_true", help="Include lamps in point-cloud export")
    ap.add_argument("--pointcloud-drop-ground-buffer", action="store_true", help="Exclude generic ground buffer meshes named like 'ground' from point-cloud export")
    ap.add_argument("--pointcloud-skip-ply", action="store_true", help="Skip writing airborne_pointcloud.ply")
    ap.add_argument("--pointcloud-skip-npz", action="store_true", help="Skip writing airborne_pointcloud.npz")
    ap.add_argument("--pointcloud-write-csv", action="store_true", help="Also write airborne_pointcloud.csv")
    ap.add_argument("--pointcloud-seed", type=int, default=42, help="Random seed for point-cloud export")

    ap.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    ap.add_argument("--verbose", action="store_true", help="Print progress messages")
    return ap



def _call_save_street_data(args, image_path: Path, cfg_path: Path):
    sig = inspect.signature(save_street_data)
    kwargs = {
        "image_path": str(image_path),
        "out_json": str(cfg_path),
        "model": args.model,
        "provider": args.provider,
        "policy": args.policy,
        "temperature": args.temperature,
        "ollama_url": args.ollama_url,
        "openai_api_key": args.openai_api_key,
        "openai_base_url": args.openai_base_url,
        "request_timeout": args.request_timeout,
        "validate_schema": not args.no_validate,
        "save_vision_json": args.save_vision,
        "save_vision_passes_dir": args.save_vision_passes_dir,
        "print_vision_summary": args.print_vision_summary,
        "print_vision_json": args.print_vision_json,
        "print_vision_passes": args.print_vision_passes,
        "verbose": args.verbose,
    }
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return save_street_data(**supported)



def _property_blend_path(outdir: Path, property_name: str) -> Path:
    return outdir / f"prop_{property_name}.blend"



def _property_image_path(outdir: Path, property_name: str, render_mode: str) -> Path:
    suffix = ".mp4" if render_mode == "render1" else ".png"
    return outdir / f"prop_{property_name}{suffix}"



def _property_script_args(args, property_name: str, out_blend: Path) -> list[object]:
    if property_name == "enclosure":
        return ["--clear", "true", "--save_as", out_blend]

    if property_name == "isovist":
        return [
            "--probe_mode", "ped",
            "--probes_per_side", "1",
            "--radius", "100",
            "--rays_iso", "360",
            "--flatten", "true",
            "--clear", "true",
            "--save_as", out_blend,
        ]

    if property_name == "svf":
        return [
            "--probe_mode", "ped",
            "--probes", "1",
            "--radius", "40",
            "--az", "48",
            "--el", "12",
            "--tick", "0.5",
            "--hits", "false",
            "--clear", "true",
            "--save_as", out_blend,
        ]

    if property_name == "shade":
        return [
            "--lat", args.lat,
            "--lon", args.lon,
            "--date", args.date,
            "--tstart", args.tstart,
            "--tend", args.tend,
            "--tz", args.tz,
            "--north", args.north,
            "--clear", "true",
            "--save_as", out_blend,
        ]

    raise ValueError(f"Unsupported property name: {property_name}")



def _run_property_visualization(
    args,
    scripts_dir: Path,
    source_blend: Path,
    outdir: Path,
    property_name: str,
) -> dict[str, str]:
    prop_script = _script_path(scripts_dir, PROPERTY_SCRIPT_NAMES[property_name])
    prop_blend = _property_blend_path(outdir, property_name)
    prop_image = _property_image_path(outdir, property_name, args.property_render_mode)

    viz_cmd = [
        args.blender,
        "-b",
        source_blend,
        "-P",
        prop_script,
        "--",
        *_property_script_args(args, property_name, prop_blend),
    ]
    run_cmd(viz_cmd, cwd=scripts_dir, dry_run=args.dry_run)
    if not args.dry_run:
        _ensure_file(prop_blend, f"{property_name} property blend")

    render_cmd = [
        args.blender,
        "-b",
        prop_blend,
        "-P",
        _script_path(scripts_dir, "05_render.py"),
        "--",
        "--out",
        prop_image,
        "--mode",
        args.property_render_mode,
        "--fps",
        args.fps,
        "--duration_s",
        args.render_duration_s,
        *_render_motion_args(args),
        *_render_resolution_args(args),
    ]
    run_cmd(render_cmd, cwd=scripts_dir, dry_run=args.dry_run)
    if not args.dry_run:
        _ensure_file(prop_image, f"{property_name} property render")

    return {
        "blend": str(prop_blend),
        "image": str(prop_image),
    }



def _run_metrics_json(
    args,
    scripts_dir: Path,
    source_blend: Path,
    outdir: Path,
) -> Path:
    out_json = outdir / args.metrics_outname
    cmd = [
        args.blender,
        "-b",
        source_blend,
        "-P",
        _script_path(scripts_dir, METRICS_SCRIPT_NAME),
        "--",
        "--out",
        out_json,
        "--lat",
        args.lat,
        "--lon",
        args.lon,
        "--date",
        args.date,
        "--tstart",
        args.tstart,
        "--tend",
        args.tend,
        "--tz",
        args.tz,
        "--north",
        args.north,
    ]
    run_cmd(cmd, cwd=scripts_dir, dry_run=args.dry_run)
    if not args.dry_run:
        _ensure_file(out_json, "street metrics JSON")
    return out_json



def _run_solweig_export(
    args,
    scripts_dir: Path,
    source_blend: Path,
    outdir: Path,
) -> dict[str, object]:
    solweig_outdir = Path(args.solweig_outdir).expanduser().resolve() if args.solweig_outdir else (outdir / "solweig_inputs")
    solweig_outdir.mkdir(parents=True, exist_ok=True)

    export_cmd = [
        args.blender,
        "-b",
        source_blend,
        "-P",
        _script_path(scripts_dir, SOLWEIG_EXPORT_SCRIPT_NAME),
        "--",
        "--outdir",
        solweig_outdir,
        "--cellsize",
        args.solweig_cellsize,
        "--padding",
        args.solweig_padding,
        "--tree-class",
        args.solweig_tree_class,
        "--ground-class",
        args.solweig_ground_class,
        "--include-vehicles",
        str(bool(args.solweig_include_vehicles)).lower(),
        "--include-humans",
        str(bool(args.solweig_include_humans)).lower(),
        "--include-lamps",
        str(bool(args.solweig_include_lamps)).lower(),
        "--keep-ground-buffer",
        str(not bool(args.solweig_drop_ground_buffer)).lower(),
        "--write-npy",
        str(bool(args.solweig_write_npy)).lower(),
        "--verbose",
        "true",
    ]
    if args.solweig_origin_x is not None:
        export_cmd.extend(["--origin-x", args.solweig_origin_x])
    if args.solweig_origin_y is not None:
        export_cmd.extend(["--origin-y", args.solweig_origin_y])
    if args.solweig_epsg is not None:
        export_cmd.extend(["--epsg", args.solweig_epsg])

    run_cmd(export_cmd, cwd=scripts_dir, dry_run=args.dry_run)

    ascii_paths = {
        "dsm": solweig_outdir / "dsm.asc",
        "dem": solweig_outdir / "dem.asc",
        "cdsm": solweig_outdir / "cdsm.asc",
        "tdsm": solweig_outdir / "tdsm.asc",
        "building_mask": solweig_outdir / "building_mask.asc",
        "landcover": solweig_outdir / "landcover.asc",
        "wall_height": solweig_outdir / "wall_height.asc",
        "wall_aspect": solweig_outdir / "wall_aspect.asc",
        "meta": solweig_outdir / "solweig_grid_meta.json",
    }
    if not args.dry_run:
        for label, path in ascii_paths.items():
            _ensure_file(path, f"SOLWEIG {label} output")

    geotiff_paths: dict[str, Path] = {}
    if not args.solweig_skip_geotiff:
        convert_cmd = [
            sys.executable,
            _script_path(scripts_dir, SOLWEIG_GEOTIFF_SCRIPT_NAME),
            "--indir",
            solweig_outdir,
        ]
        if args.solweig_epsg is not None:
            convert_cmd.extend(["--epsg", args.solweig_epsg])
        run_cmd(convert_cmd, cwd=scripts_dir, dry_run=args.dry_run)

        geotiff_paths = {
            "dsm": solweig_outdir / "dsm.tif",
            "dem": solweig_outdir / "dem.tif",
            "cdsm": solweig_outdir / "cdsm.tif",
            "tdsm": solweig_outdir / "tdsm.tif",
            "building_mask": solweig_outdir / "building_mask.tif",
            "landcover": solweig_outdir / "landcover.tif",
            "wall_height": solweig_outdir / "wall_height.tif",
            "wall_aspect": solweig_outdir / "wall_aspect.tif",
        }
        if not args.dry_run:
            for label, path in geotiff_paths.items():
                _ensure_file(path, f"SOLWEIG {label} GeoTIFF")

    return {
        "outdir": str(solweig_outdir),
        "ascii": {k: str(v) for k, v in ascii_paths.items()},
        "geotiff": {k: str(v) for k, v in geotiff_paths.items()},
    }



def _run_envimet_export(
    args,
    scripts_dir: Path,
    source_blend: Path,
    outdir: Path,
) -> dict[str, object]:
    envimet_outdir = Path(args.envimet_outdir).expanduser().resolve() if args.envimet_outdir else (outdir / "envimet_voxels")
    envimet_outdir.mkdir(parents=True, exist_ok=True)

    export_cmd = [
        args.blender,
        "-b",
        source_blend,
        "-P",
        _script_path(scripts_dir, ENVIMET_EXPORT_SCRIPT_NAME),
        "--",
        "--outdir",
        envimet_outdir,
        "--dx",
        args.envimet_dx,
        "--dy",
        args.envimet_dy,
        "--dz",
        args.envimet_dz,
        "--padding",
        args.envimet_padding,
        "--tree-class",
        args.envimet_tree_class,
        "--ground-class",
        args.envimet_ground_class,
        "--include-vehicles",
        str(bool(args.envimet_include_vehicles)).lower(),
        "--include-humans",
        str(bool(args.envimet_include_humans)).lower(),
        "--include-lamps",
        str(bool(args.envimet_include_lamps)).lower(),
        "--keep-ground-buffer",
        str(not bool(args.envimet_drop_ground_buffer)).lower(),
        "--canopy-min-height",
        args.envimet_canopy_min_height,
        "--woody-min-height",
        args.envimet_woody_min_height,
        "--write-npz",
        str(not bool(args.envimet_skip_npz)).lower(),
        "--write-json-fallback",
        str(not bool(args.envimet_skip_json_fallback)).lower(),
        "--verbose",
        str(bool(args.verbose)).lower(),
    ]
    if args.envimet_origin_x is not None:
        export_cmd.extend(["--origin-x", args.envimet_origin_x])
    if args.envimet_origin_y is not None:
        export_cmd.extend(["--origin-y", args.envimet_origin_y])
    if args.envimet_epsg is not None:
        export_cmd.extend(["--epsg", args.envimet_epsg])

    run_cmd(export_cmd, cwd=scripts_dir, dry_run=args.dry_run)

    meta_path = envimet_outdir / "envimet_voxel_meta.json"
    bundle_path = envimet_outdir / "envimet_voxels.npz"
    if not args.dry_run:
        _ensure_file(meta_path, "ENVI-met voxel metadata")
        if not args.envimet_skip_npz:
            _ensure_file(bundle_path, "ENVI-met voxel NPZ bundle")

    return {
        "outdir": str(envimet_outdir),
        "meta": str(meta_path),
        "bundle": None if args.envimet_skip_npz else str(bundle_path),
    }


def _run_pointcloud_export(
    args,
    scripts_dir: Path,
    source_blend: Path,
    outdir: Path,
) -> dict[str, object]:
    pointcloud_outdir = Path(args.pointcloud_outdir).expanduser().resolve() if args.pointcloud_outdir else (outdir / "pointcloud")
    pointcloud_outdir.mkdir(parents=True, exist_ok=True)

    export_cmd = [
        args.blender,
        "-b",
        source_blend,
        "-P",
        _script_path(scripts_dir, POINTCLOUD_EXPORT_SCRIPT_NAME),
        "--",
        "--outdir",
        pointcloud_outdir,
        "--spacing",
        args.pointcloud_spacing,
        "--padding",
        args.pointcloud_padding,
        "--scan-count",
        args.pointcloud_scan_count,
        "--jitter-frac",
        args.pointcloud_jitter_frac,
        "--noise-xy",
        args.pointcloud_noise_xy,
        "--noise-z",
        args.pointcloud_noise_z,
        "--dropout",
        args.pointcloud_dropout,
        "--ground-return-prob",
        args.pointcloud_ground_return_prob,
        "--include-vehicles",
        str(bool(args.pointcloud_include_vehicles)).lower(),
        "--include-humans",
        str(bool(args.pointcloud_include_humans)).lower(),
        "--include-lamps",
        str(bool(args.pointcloud_include_lamps)).lower(),
        "--keep-ground-buffer",
        str(not bool(args.pointcloud_drop_ground_buffer)).lower(),
        "--write-ply",
        str(not bool(args.pointcloud_skip_ply)).lower(),
        "--write-npz",
        str(not bool(args.pointcloud_skip_npz)).lower(),
        "--write-csv",
        str(bool(args.pointcloud_write_csv)).lower(),
        "--seed",
        args.pointcloud_seed,
        "--verbose",
        str(bool(args.verbose)).lower(),
    ]
    if args.pointcloud_origin_x is not None:
        export_cmd.extend(["--origin-x", args.pointcloud_origin_x])
    if args.pointcloud_origin_y is not None:
        export_cmd.extend(["--origin-y", args.pointcloud_origin_y])
    if args.pointcloud_epsg is not None:
        export_cmd.extend(["--epsg", args.pointcloud_epsg])

    run_cmd(export_cmd, cwd=scripts_dir, dry_run=args.dry_run)

    meta_path = pointcloud_outdir / "pointcloud_meta.json"
    ply_path = pointcloud_outdir / "airborne_pointcloud.ply"
    npz_path = pointcloud_outdir / "airborne_pointcloud.npz"
    csv_path = pointcloud_outdir / "airborne_pointcloud.csv"

    if not args.dry_run:
        _ensure_file(meta_path, "point-cloud metadata")
        if not args.pointcloud_skip_ply:
            _ensure_file(ply_path, "point-cloud PLY")
        if not args.pointcloud_skip_npz:
            _ensure_file(npz_path, "point-cloud NPZ")
        if args.pointcloud_write_csv:
            _ensure_file(csv_path, "point-cloud CSV")

    return {
        "outdir": str(pointcloud_outdir),
        "meta": str(meta_path),
        "ply": None if args.pointcloud_skip_ply else str(ply_path),
        "npz": None if args.pointcloud_skip_npz else str(npz_path),
        "csv": None if not args.pointcloud_write_csv else str(csv_path),
    }


def main() -> int:
    args = build_parser().parse_args()

    image_path = Path(args.image).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    scripts_dir = Path(args.scripts_dir).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve() if args.assets_dir else scripts_dir

    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        return 1
    if not scripts_dir.exists():
        print(f"Error: scripts dir not found: {scripts_dir}", file=sys.stderr)
        return 1
    if not assets_dir.exists():
        print(f"Error: assets dir not found: {assets_dir}", file=sys.stderr)
        return 1

    property_names = _csv_to_property_list(args.properties)

    required_scripts = ["01_model.py", "02_textured.py", "03_animated.py", "04_soundscapes.py", "05_render.py"]
    if not args.skip_property_images:
        required_scripts.extend(PROPERTY_SCRIPT_NAMES[name] for name in property_names)
    if not args.skip_metrics_json:
        required_scripts.append(METRICS_SCRIPT_NAME)
    if not args.skip_solweig_export:
        required_scripts.append(SOLWEIG_EXPORT_SCRIPT_NAME)
        if not args.solweig_skip_geotiff:
            required_scripts.append(SOLWEIG_GEOTIFF_SCRIPT_NAME)
    if not args.skip_envimet_export:
        required_scripts.append(ENVIMET_EXPORT_SCRIPT_NAME)
    if not args.skip_pointcloud_export:
        required_scripts.append(POINTCLOUD_EXPORT_SCRIPT_NAME)

    missing = [name for name in required_scripts if _find_script(scripts_dir, name) is None]
    if missing:
        print(f"Error: missing scripts in {scripts_dir}: {', '.join(missing)}", file=sys.stderr)
        return 1

    outdir.mkdir(parents=True, exist_ok=True)

    cfg_path = outdir / "street_data.json"
    model_blend = outdir / "01_model.blend"
    textured_blend = outdir / "02_textured.blend"
    animated_blend = outdir / "03_animated.blend"
    sounded_blend = outdir / "04_sounded.blend"
    final_output = outdir / _render_output_name(args.render_mode, args.render_outname)

    try:
        if args.verbose:
            print("[pipeline] Step 1/6: image -> street_data.json", flush=True)
        _call_save_street_data(args, image_path, cfg_path)
        if args.verbose:
            print("[pipeline] street_data.json created.", flush=True)
        if not args.dry_run:
            _ensure_file(cfg_path, "street_data.json")

        if args.verbose:
            print("[pipeline] Step 2/6: 01_model.py", flush=True)
        run_cmd(
            [
                args.blender,
                "-b",
                "-P",
                _script_path(scripts_dir, "01_model.py"),
                "--",
                "--config",
                cfg_path,
                "--outblend",
                model_blend,
                "--asset-root",
                assets_dir,
            ],
            cwd=scripts_dir,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            _ensure_file(model_blend, "01_model.blend")

        current_blend = model_blend

        if not args.skip_texture:
            if args.verbose:
                print("[pipeline] Step 3/6: 02_textured.py", flush=True)
            run_cmd(
                [
                    args.blender,
                    "-b",
                    current_blend,
                    "-P",
                    _script_path(scripts_dir, "02_textured.py"),
                    "--",
                    "--outblend",
                    textured_blend,
                    "--texdir",
                    assets_dir / "textures",
                    "--pack",
                    "true",
                    "--relative",
                    "true",
                ],
                cwd=scripts_dir,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                _ensure_file(textured_blend, "02_textured.blend")
            current_blend = textured_blend
        elif args.verbose:
            print("[pipeline] Skipping texture stage.", flush=True)

        if not args.skip_animate:
            if args.verbose:
                print("[pipeline] Step 4/6: 03_animated.py", flush=True)
            run_cmd(
                [
                    args.blender,
                    "-b",
                    current_blend,
                    "-P",
                    _script_path(scripts_dir, "03_animated.py"),
                    "--",
                    "--fps",
                    args.fps,
                    "--duration",
                    args.duration,
                    "--mixamo_dir",
                    assets_dir / "mixamo_fbx",
                    "--out",
                    animated_blend,
                ],
                cwd=scripts_dir,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                _ensure_file(animated_blend, "03_animated.blend")
            current_blend = animated_blend
        elif args.verbose:
            print("[pipeline] Skipping animation stage.", flush=True)

        if not args.skip_sound:
            if args.verbose:
                print("[pipeline] Step 5/6: 04_soundscapes.py", flush=True)
            run_cmd(
                [
                    args.blender,
                    "-b",
                    current_blend,
                    "-P",
                    _script_path(scripts_dir, "04_soundscapes.py"),
                    "--",
                    "--outblend",
                    sounded_blend,
                    "--dir_car",
                    assets_dir / "sounds" / "car",
                    "--dir_walk",
                    assets_dir / "sounds" / "walk",
                    "--dir_run",
                    assets_dir / "sounds" / "run",
                    "--dir_bird",
                    assets_dir / "sounds" / "bird",
                    "--dir_wind",
                    assets_dir / "sounds" / "wind",
                    "--dir_amb",
                    assets_dir / "sounds" / "amb",
                ],
                cwd=scripts_dir,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                _ensure_file(sounded_blend, "04_sounded.blend")
            current_blend = sounded_blend
        elif args.verbose:
            print("[pipeline] Skipping sound stage.", flush=True)

        if not args.skip_render:
            if args.verbose:
                print("[pipeline] Step 6/6: 05_render.py", flush=True)
            run_cmd(
                [
                    args.blender,
                    "-b",
                    current_blend,
                    "-P",
                    _script_path(scripts_dir, "05_render.py"),
                    "--",
                    "--out",
                    final_output,
                    "--mode",
                    args.render_mode,
                    "--fps",
                    args.fps,
                    "--duration_s",
                    args.render_duration_s,
                    *_render_motion_args(args),
                    *_render_resolution_args(args),
                ],
                cwd=scripts_dir,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                _ensure_file(final_output, "final render output")
        elif args.verbose:
            print("[pipeline] Skipping render stage.", flush=True)

        # Property, metrics, and export stages use the latest blend produced above.
        property_outputs: dict[str, dict[str, str]] = {}
        metrics_json: Optional[Path] = None
        solweig_outputs: Optional[dict[str, object]] = None
        envimet_outputs: Optional[dict[str, object]] = None
        pointcloud_outputs: Optional[dict[str, object]] = None

        if not args.skip_property_images:
            if args.verbose:
                print("[pipeline] Property stage: generating property visualization blends and images", flush=True)
            for property_name in property_names:
                property_outputs[property_name] = _run_property_visualization(
                    args=args,
                    scripts_dir=scripts_dir,
                    source_blend=current_blend,
                    outdir=outdir,
                    property_name=property_name,
                )

        if not args.skip_metrics_json:
            if args.verbose:
                print("[pipeline] Metrics stage: generating street_metrics.json", flush=True)
            metrics_json = _run_metrics_json(
                args=args,
                scripts_dir=scripts_dir,
                source_blend=current_blend,
                outdir=outdir,
            )

        if not args.skip_solweig_export:
            if args.verbose:
                print("[pipeline] SOLWEIG stage: exporting DSM/DEM/CDSM/TDSM/landcover rasters", flush=True)
            solweig_outputs = _run_solweig_export(
                args=args,
                scripts_dir=scripts_dir,
                source_blend=current_blend,
                outdir=outdir,
            )

        if not args.skip_envimet_export:
            if args.verbose:
                print("[pipeline] ENVI-met stage: exporting voxel grids", flush=True)
            envimet_outputs = _run_envimet_export(
                args=args,
                scripts_dir=scripts_dir,
                source_blend=current_blend,
                outdir=outdir,
            )

        if not args.skip_pointcloud_export:
            if args.verbose:
                print("[pipeline] Point-cloud stage: exporting airborne semantic point cloud", flush=True)
            pointcloud_outputs = _run_pointcloud_export(
                args=args,
                scripts_dir=scripts_dir,
                source_blend=current_blend,
                outdir=outdir,
            )

        print("\nPipeline completed.", flush=True)
        print(f"street_data.json: {cfg_path}", flush=True)
        print(f"01_model.blend:  {model_blend}", flush=True)
        if not args.skip_texture:
            print(f"02_textured.blend: {textured_blend}", flush=True)
        if not args.skip_animate:
            print(f"03_animated.blend: {animated_blend}", flush=True)
        if not args.skip_sound:
            print(f"04_sounded.blend: {sounded_blend}", flush=True)
        if not args.skip_render:
            print(f"final output: {final_output}", flush=True)
        if property_outputs:
            for property_name, outputs in property_outputs.items():
                print(f"{property_name} blend: {outputs['blend']}", flush=True)
                print(f"{property_name} image: {outputs['image']}", flush=True)
        if metrics_json is not None:
            print(f"metrics JSON: {metrics_json}", flush=True)
        if solweig_outputs is not None:
            print(f"SOLWEIG outdir: {solweig_outputs['outdir']}", flush=True)
            for label, path in solweig_outputs.get("ascii", {}).items():
                print(f"SOLWEIG {label}: {path}", flush=True)
            for label, path in solweig_outputs.get("geotiff", {}).items():
                print(f"SOLWEIG {label} GeoTIFF: {path}", flush=True)
        if envimet_outputs is not None:
            print(f"ENVI-met outdir: {envimet_outputs['outdir']}", flush=True)
            print(f"ENVI-met metadata: {envimet_outputs['meta']}", flush=True)
            if envimet_outputs.get("bundle"):
                print(f"ENVI-met voxel bundle: {envimet_outputs['bundle']}", flush=True)
        if pointcloud_outputs is not None:
            print(f"Point-cloud outdir: {pointcloud_outputs['outdir']}", flush=True)
            print(f"Point-cloud metadata: {pointcloud_outputs['meta']}", flush=True)
            if pointcloud_outputs.get("ply"):
                print(f"Point-cloud PLY: {pointcloud_outputs['ply']}", flush=True)
            if pointcloud_outputs.get("npz"):
                print(f"Point-cloud NPZ: {pointcloud_outputs['npz']}", flush=True)
            if pointcloud_outputs.get("csv"):
                print(f"Point-cloud CSV: {pointcloud_outputs['csv']}", flush=True)

        summary = {
            "street_data_json": str(cfg_path),
            "model_blend": str(model_blend),
            "textured_blend": None if args.skip_texture else str(textured_blend),
            "animated_blend": None if args.skip_animate else str(animated_blend),
            "sounded_blend": None if args.skip_sound else str(sounded_blend),
            "latest_pipeline_blend": str(current_blend),
            "final_output": None if args.skip_render else str(final_output),
            "render_mode": args.render_mode,
            "render_pan_deg": args.render_pan_deg,
            "render_pan_center_deg": args.render_pan_center_deg,
            "render_rotations": args.render_rotations,
            "render_exposure": args.render_exposure,
            "property_render_mode": None if args.skip_property_images else args.property_render_mode,
            "property_outputs": property_outputs,
            "metrics_json": None if metrics_json is None else str(metrics_json),
            "solweig_outputs": solweig_outputs,
            "envimet_outputs": envimet_outputs,
            "pointcloud_outputs": pointcloud_outputs,
        }
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    except subprocess.CalledProcessError as exc:
        print(f"Error: stage failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
