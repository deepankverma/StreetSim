#!/usr/bin/env python3
"""Convert ESRI ASCII grids from solweig_export_rasters.py into GeoTIFF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


ASC_NAMES = [
    "dsm",
    "dem",
    "cdsm",
    "tdsm",
    "building_mask",
    "landcover",
    "wall_height",
    "wall_aspect",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert exported SOLWEIG ASCII rasters to GeoTIFF.")
    ap.add_argument("--indir", required=True, help="Folder containing *.asc and solweig_grid_meta.json")
    ap.add_argument("--outdir", help="Optional output folder. Default: same as --indir")
    ap.add_argument("--epsg", type=int, help="Override EPSG code if not present in metadata")
    return ap.parse_args()


def read_ascii_grid(path: Path):
    header = {}
    with path.open("r", encoding="utf-8") as f:
        for _ in range(6):
            k, v = f.readline().split(None, 1)
            header[k.lower()] = float(v.strip())
    data = np.loadtxt(path, skiprows=6)
    if data.ndim == 1:
        data = data.reshape((1, -1))
    return header, data


def main() -> int:
    args = parse_args()
    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)

    meta_path = indir / "solweig_grid_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    epsg = args.epsg or meta.get("grid", {}).get("epsg")

    for stem in ASC_NAMES:
        asc_path = indir / f"{stem}.asc"
        if not asc_path.exists():
            continue
        header, data = read_ascii_grid(asc_path)
        transform = from_origin(
            west=header["xllcorner"],
            north=header["yllcorner"] + header["nrows"] * header["cellsize"],
            xsize=header["cellsize"],
            ysize=header["cellsize"],
        )
        out_path = outdir / f"{stem}.tif"
        profile = {
            "driver": "GTiff",
            "height": int(header["nrows"]),
            "width": int(header["ncols"]),
            "count": 1,
            "dtype": data.dtype,
            "transform": transform,
            "nodata": header.get("nodata_value"),
        }
        if epsg:
            profile["crs"] = f"EPSG:{int(epsg)}"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
