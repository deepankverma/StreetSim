#!/usr/bin/env python3
"""Export SOLWEIG-style rasters from the current Blender scene.

Designed for the street pipeline in this project. It rasterizes the active .blend
into local-coordinate grids that can later be converted to GeoTIFF and used as
inputs for SVF, wall metrics, and SOLWEIG-style thermal workflows.

Outputs (ESRI ASCII grids by default):
- dsm.asc           : top surface elevation (m)
- dem.asc           : ground / bare-earth style elevation (m)
- cdsm.asc          : canopy height above ground (m)
- tdsm.asc          : trunk / woody vegetation height above ground (m)
- building_mask.asc : 1 where building/roof is the top static surface, else 0
- landcover.asc     : UMEP-style land-cover codes
- wall_height.asc   : derived wall height raster (m)
- wall_aspect.asc   : derived wall aspect raster (deg clockwise from north)
- solweig_grid_meta.json

Notes:
- Coordinates are in the Blender scene coordinate system unless --origin-x/
  --origin-y are supplied to anchor the lower-left corner in projected space.
- The script intentionally ignores cameras, lights, speakers, empties, and by
  default also vehicles and humans so the geometry export stays stable even
  when run on 03_animated.blend or 04_sounded.blend.
- wall_height/wall_aspect are derived from the exported DSM/DEM/building mask.
  They are intended as a practical SOLWEIG preprocessor output, not as an exact
  reimplementation of UMEP's wall preprocessor.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Vector


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_bool(text: str | bool | None, default: bool = False) -> bool:
    if text is None:
        return default
    if isinstance(text, bool):
        return text
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> dict:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    args = {
        "outdir": None,
        "cellsize": 1.0,
        "padding": 2.0,
        "z_margin": 25.0,
        "origin_x": None,
        "origin_y": None,
        "epsg": None,
        "tree_class": "deciduous",   # deciduous|evergreen
        "ground_class": "bare_soil", # grass|bare_soil|paved
        "include_vehicles": False,
        "include_humans": False,
        "include_lamps": False,
        "keep_ground_buffer": True,
        "cdsm_min_height": 0.5,
        "tdsm_min_height": 0.5,
        "wall_min_height": 0.5,
        "nodata": -9999.0,
        "write_npy": False,
        "verbose": True,
    }

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--outdir" and i + 1 < len(argv):
            args["outdir"] = argv[i + 1]
            i += 2
            continue
        if a == "--cellsize" and i + 1 < len(argv):
            args["cellsize"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--padding" and i + 1 < len(argv):
            args["padding"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--z-margin" and i + 1 < len(argv):
            args["z_margin"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--origin-x" and i + 1 < len(argv):
            args["origin_x"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--origin-y" and i + 1 < len(argv):
            args["origin_y"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--epsg" and i + 1 < len(argv):
            args["epsg"] = int(argv[i + 1])
            i += 2
            continue
        if a == "--tree-class" and i + 1 < len(argv):
            args["tree_class"] = str(argv[i + 1]).strip().lower()
            i += 2
            continue
        if a == "--ground-class" and i + 1 < len(argv):
            args["ground_class"] = str(argv[i + 1]).strip().lower()
            i += 2
            continue
        if a == "--include-vehicles" and i + 1 < len(argv):
            args["include_vehicles"] = _parse_bool(argv[i + 1], False)
            i += 2
            continue
        if a == "--include-humans" and i + 1 < len(argv):
            args["include_humans"] = _parse_bool(argv[i + 1], False)
            i += 2
            continue
        if a == "--include-lamps" and i + 1 < len(argv):
            args["include_lamps"] = _parse_bool(argv[i + 1], False)
            i += 2
            continue
        if a == "--keep-ground-buffer" and i + 1 < len(argv):
            args["keep_ground_buffer"] = _parse_bool(argv[i + 1], True)
            i += 2
            continue
        if a == "--cdsm-min-height" and i + 1 < len(argv):
            args["cdsm_min_height"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--tdsm-min-height" and i + 1 < len(argv):
            args["tdsm_min_height"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--wall-min-height" and i + 1 < len(argv):
            args["wall_min_height"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--nodata" and i + 1 < len(argv):
            args["nodata"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--write-npy" and i + 1 < len(argv):
            args["write_npy"] = _parse_bool(argv[i + 1], False)
            i += 2
            continue
        if a == "--verbose" and i + 1 < len(argv):
            args["verbose"] = _parse_bool(argv[i + 1], True)
            i += 2
            continue
        i += 1

    if not args["outdir"]:
        raise SystemExit("Error: --outdir is required")
    return args


# -----------------------------------------------------------------------------
# Scene helpers
# -----------------------------------------------------------------------------

def dg():
    return bpy.context.evaluated_depsgraph_get()


def _world_bbox(obj) -> tuple[Vector, Vector]:
    eo = obj.evaluated_get(dg())
    mat = eo.matrix_world
    corners = [mat @ Vector(c) for c in eo.bound_box]
    mn = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    mx = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return mn, mx


def _log(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg, flush=True)


def _name(obj) -> str:
    return obj.name.lower() if obj else ""


def is_mesh(obj) -> bool:
    return bool(obj) and obj.type == "MESH" and not getattr(obj, "hide_render", False)


def is_vehicle(obj) -> bool:
    nm = _name(obj)
    return any(t in nm for t in (
        "parkedcar", "car_", "car.", " car", "vehicle", "truck", "bus", "van",
        "motorcycle", "scooter", "bicycle", "bike_",
    ))


def is_human(obj) -> bool:
    nm = _name(obj)
    return any(t in nm for t in ("human", "person", "pedestrian"))


def is_lamp(obj) -> bool:
    nm = _name(obj)
    return any(t in nm for t in ("lamp", "lightpole", "streetlight"))


def is_building(obj) -> bool:
    nm = _name(obj)
    return any(t in nm for t in ("building", "roof"))


def is_canopy(obj) -> bool:
    nm = _name(obj)
    return any(t in nm for t in ("leaf", "leaves", "canopy", "foliage"))


def is_tree_woody(obj) -> bool:
    nm = _name(obj)
    if is_canopy(obj):
        return False
    return any(t in nm for t in ("_wood", "trunk", "branch", "stem", "bark")) or nm.startswith("tree_")


def is_low_vegetation(obj) -> bool:
    nm = _name(obj)
    return any(t in nm for t in ("shrub", "grass", "vegetation", "plant"))


def is_veg(obj) -> bool:
    return is_canopy(obj) or is_tree_woody(obj) or is_low_vegetation(obj)


def is_ground(obj, keep_ground_buffer: bool = True) -> bool:
    nm = _name(obj)
    ground_tokens = (
        "driveway", "bikepath", "footpath", "median", "street_parking", "parking", "gutter",
        "ground", "plaza", "pavement", "sidewalk", "walkway", "road", "lane", "asphalt", "curb",
    )
    if not any(t in nm for t in ground_tokens):
        return False
    if not keep_ground_buffer and nm == "ground":
        return False
    return True


def ground_landcover_code(obj, ground_class: str) -> int:
    nm = _name(obj)
    if any(t in nm for t in ("driveway", "bikepath", "footpath", "parking", "gutter", "road", "lane", "asphalt", "pavement")):
        return 1  # paved
    if "median" in nm:
        return 1  # keep medians paved unless vegetation overrides via CDSM
    if "ground" in nm:
        if ground_class == "grass":
            return 5
        if ground_class == "paved":
            return 1
        return 6
    return 6


@dataclass
class Hit:
    ok: bool
    location: Vector | None = None
    obj: bpy.types.Object | None = None


class SceneClassifier:
    def __init__(self, *, include_vehicles: bool, include_humans: bool, include_lamps: bool, keep_ground_buffer: bool):
        self.include_vehicles = include_vehicles
        self.include_humans = include_humans
        self.include_lamps = include_lamps
        self.keep_ground_buffer = keep_ground_buffer
        self._deps = dg()
        self._scene = bpy.context.scene

    def is_static_mesh(self, obj) -> bool:
        if not is_mesh(obj):
            return False
        if not self.include_vehicles and is_vehicle(obj):
            return False
        if not self.include_humans and is_human(obj):
            return False
        if not self.include_lamps and is_lamp(obj):
            return False
        return True

    def extent_objects(self):
        out = []
        for obj in bpy.data.objects:
            if not self.is_static_mesh(obj):
                continue
            if is_ground(obj, self.keep_ground_buffer) or is_building(obj) or is_veg(obj):
                out.append(obj)
        return out

    def cast_down_until(self, origin: Vector, predicate, max_dist: float, max_hops: int = 32, eps: float = 1e-4) -> Hit:
        start = origin.copy()
        remaining = float(max_dist)
        for _ in range(max_hops):
            hit, loc, _norm, _face_idx, obj, _mat = self._scene.ray_cast(self._deps, start, Vector((0, 0, -1)), distance=remaining)
            if not hit:
                return Hit(False, None, None)
            if self.is_static_mesh(obj) and predicate(obj):
                return Hit(True, loc, obj)
            step = max((start.z - loc.z) + eps, eps)
            start = Vector((start.x, start.y, loc.z - eps))
            remaining -= step
            if remaining <= 0:
                break
        return Hit(False, None, None)


# -----------------------------------------------------------------------------
# Raster helpers
# -----------------------------------------------------------------------------

def _fmt_value(v: float) -> str:
    if abs(v - round(v)) < 1e-10:
        return str(int(round(v)))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def write_ascii_grid(path: Path, grid: list[list[float]], *, xllcorner: float, yllcorner: float, cellsize: float, nodata: float) -> None:
    nrows = len(grid)
    ncols = len(grid[0]) if nrows else 0
    with path.open("w", encoding="utf-8") as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {_fmt_value(xllcorner)}\n")
        f.write(f"yllcorner     {_fmt_value(yllcorner)}\n")
        f.write(f"cellsize      {_fmt_value(cellsize)}\n")
        f.write(f"NODATA_value  {_fmt_value(nodata)}\n")
        for row in grid:
            f.write(" ".join(_fmt_value(v) for v in row))
            f.write("\n")


def _derive_wall_rasters(
    dsm: list[list[float]],
    dem: list[list[float]],
    building_mask: list[list[float]],
    *,
    nodata: float,
    wall_min_height: float,
) -> tuple[list[list[float]], list[list[float]], dict[str, float]]:
    """Derive practical wall-height and wall-aspect rasters from exported grids.

    Strategy:
    - treat each non-building cell adjacent to at least one building cell as a wall pixel
    - wall height = max neighboring building top above local ground
    - wall aspect = outward facing direction from neighboring building cell to wall pixel,
      expressed as degrees clockwise from north (+Y = 0°)
    - non-wall pixels remain 0 in both rasters

    This approximates UMEP's wall preprocessing in a way that is robust for the
    synthetic street scenes produced here.
    """
    nrows = len(dsm)
    ncols = len(dsm[0]) if nrows else 0
    wall_height = [[0.0 for _ in range(ncols)] for _ in range(nrows)]
    wall_aspect = [[0.0 for _ in range(ncols)] for _ in range(nrows)]

    wall_pixels = 0
    max_wall = 0.0

    for r in range(nrows):
        for c in range(ncols):
            if building_mask[r][c] > 0.5:
                continue
            ground_z = dem[r][c]
            if ground_z == nodata:
                continue

            best_h = 0.0
            best_aspect = 0.0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                        continue
                    if building_mask[rr][cc] <= 0.5:
                        continue
                    neigh_top = dsm[rr][cc]
                    if neigh_top == nodata:
                        continue

                    h = max(0.0, float(neigh_top - ground_z))
                    if h <= best_h:
                        continue

                    # Convert raster offset to a compass aspect.
                    # +Y is north in this project; raster row index increases southward.
                    dx = float(dc)
                    dy = float(-dr)
                    aspect = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
                    if aspect == 0.0:
                        aspect = 360.0
                    best_h = h
                    best_aspect = aspect

            if best_h >= wall_min_height:
                wall_height[r][c] = best_h
                wall_aspect[r][c] = best_aspect
                wall_pixels += 1
                max_wall = max(max_wall, best_h)

    return wall_height, wall_aspect, {
        "wall_pixel_count": wall_pixels,
        "max_wall_height_m": max_wall,
    }


# -----------------------------------------------------------------------------
# Main export
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    outdir = Path(args["outdir"]).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    classifier = SceneClassifier(
        include_vehicles=args["include_vehicles"],
        include_humans=args["include_humans"],
        include_lamps=args["include_lamps"],
        keep_ground_buffer=args["keep_ground_buffer"],
    )

    extent_objs = classifier.extent_objects()
    if not extent_objs:
        raise SystemExit("Error: no eligible meshes found for raster export.")

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    for obj in extent_objs:
        mn, mx = _world_bbox(obj)
        min_x = min(min_x, mn.x)
        min_y = min(min_y, mn.y)
        min_z = min(min_z, mn.z)
        max_x = max(max_x, mx.x)
        max_y = max(max_y, mx.y)
        max_z = max(max_z, mx.z)

    padding = max(0.0, float(args["padding"]))
    cell = max(0.05, float(args["cellsize"]))

    x0_local = min_x - padding
    y0_local = min_y - padding
    x1_local = max_x + padding
    y1_local = max_y + padding

    ncols = max(1, int(math.ceil((x1_local - x0_local) / cell)))
    nrows = max(1, int(math.ceil((y1_local - y0_local) / cell)))

    # Re-fit upper bounds to the final integer grid.
    x1_local = x0_local + ncols * cell
    y1_local = y0_local + nrows * cell

    z_top = max_z + max(5.0, float(args["z_margin"]))
    max_dist = (z_top - min_z) + max(10.0, float(args["z_margin"]))

    _log(f"[SOLWEIG] Raster extent local: x=({x0_local:.3f},{x1_local:.3f}) y=({y0_local:.3f},{y1_local:.3f})", args["verbose"])
    _log(f"[SOLWEIG] Grid: {ncols} cols x {nrows} rows @ {cell:.3f} m", args["verbose"])

    nodata = float(args["nodata"])
    tree_code = 4 if args["tree_class"] != "evergreen" else 3

    dsm = [[nodata for _ in range(ncols)] for _ in range(nrows)]
    dem = [[nodata for _ in range(ncols)] for _ in range(nrows)]
    cdsm = [[0.0 for _ in range(ncols)] for _ in range(nrows)]
    tdsm = [[0.0 for _ in range(ncols)] for _ in range(nrows)]
    building_mask = [[0.0 for _ in range(ncols)] for _ in range(nrows)]
    landcover = [[nodata for _ in range(ncols)] for _ in range(nrows)]

    for row in range(nrows):
        y = y1_local - (row + 0.5) * cell
        if args["verbose"] and row % max(1, nrows // 10) == 0:
            _log(f"[SOLWEIG] Row {row + 1}/{nrows}", True)

        for col in range(ncols):
            x = x0_local + (col + 0.5) * cell
            origin = Vector((x, y, z_top))

            dsm_hit = classifier.cast_down_until(
                origin,
                lambda o: classifier.is_static_mesh(o),
                max_dist=max_dist,
            )
            dem_hit = classifier.cast_down_until(
                origin,
                lambda o: is_ground(o, classifier.keep_ground_buffer),
                max_dist=max_dist,
            )
            canopy_hit = classifier.cast_down_until(
                origin,
                lambda o: is_canopy(o) or is_low_vegetation(o),
                max_dist=max_dist,
            )
            woody_hit = classifier.cast_down_until(
                origin,
                lambda o: is_tree_woody(o),
                max_dist=max_dist,
            )
            building_hit = classifier.cast_down_until(
                origin,
                lambda o: is_building(o),
                max_dist=max_dist,
            )

            dem_z = dem_hit.location.z if dem_hit.ok else min_z
            if dsm_hit.ok:
                dsm[row][col] = float(dsm_hit.location.z)
            if dem_hit.ok:
                dem[row][col] = float(dem_z)
            else:
                dem[row][col] = float(min_z)

            if building_hit.ok and dsm_hit.ok and dsm_hit.obj == building_hit.obj:
                building_mask[row][col] = 1.0

            if canopy_hit.ok:
                canopy_h = max(0.0, float(canopy_hit.location.z - dem_z))
                if canopy_h >= float(args["cdsm_min_height"]):
                    cdsm[row][col] = canopy_h

            if woody_hit.ok:
                trunk_h = max(0.0, float(woody_hit.location.z - dem_z))
                if trunk_h >= float(args["tdsm_min_height"]):
                    tdsm[row][col] = trunk_h

            # UMEP-style land cover coding.
            if building_mask[row][col] > 0.5:
                landcover[row][col] = 2.0
            elif cdsm[row][col] > 0.0:
                landcover[row][col] = float(tree_code)
            elif dem_hit.ok:
                landcover[row][col] = float(ground_landcover_code(dem_hit.obj, args["ground_class"]))
            else:
                landcover[row][col] = float(ground_landcover_code(None, args["ground_class"]))

    wall_height, wall_aspect, wall_stats = _derive_wall_rasters(
        dsm,
        dem,
        building_mask,
        nodata=nodata,
        wall_min_height=float(args["wall_min_height"]),
    )

    xllcorner_geo = float(args["origin_x"]) if args["origin_x"] is not None else x0_local
    yllcorner_geo = float(args["origin_y"]) if args["origin_y"] is not None else y0_local

    write_ascii_grid(outdir / "dsm.asc", dsm, xllcorner=xllcorner_geo, yllcorner=yllcorner_geo, cellsize=cell, nodata=nodata)
    write_ascii_grid(outdir / "dem.asc", dem, xllcorner=xllcorner_geo, yllcorner=yllcorner_geo, cellsize=cell, nodata=nodata)
    write_ascii_grid(outdir / "cdsm.asc", cdsm, xllcorner=xllcorner_geo, yllcorner=yllcorner_geo, cellsize=cell, nodata=nodata)
    write_ascii_grid(outdir / "tdsm.asc", tdsm, xllcorner=xllcorner_geo, yllcorner=yllcorner_geo, cellsize=cell, nodata=nodata)
    write_ascii_grid(outdir / "building_mask.asc", building_mask, xllcorner=xllcorner_geo, yllcorner=yllcorner_geo, cellsize=cell, nodata=0.0)
    write_ascii_grid(outdir / "landcover.asc", landcover, xllcorner=xllcorner_geo, yllcorner=yllcorner_geo, cellsize=cell, nodata=nodata)
    write_ascii_grid(outdir / "wall_height.asc", wall_height, xllcorner=xllcorner_geo, yllcorner=yllcorner_geo, cellsize=cell, nodata=0.0)
    write_ascii_grid(outdir / "wall_aspect.asc", wall_aspect, xllcorner=xllcorner_geo, yllcorner=yllcorner_geo, cellsize=cell, nodata=0.0)

    if args["write_npy"]:
        import numpy as np

        np.save(outdir / "dsm.npy", np.array(dsm, dtype=float))
        np.save(outdir / "dem.npy", np.array(dem, dtype=float))
        np.save(outdir / "cdsm.npy", np.array(cdsm, dtype=float))
        np.save(outdir / "tdsm.npy", np.array(tdsm, dtype=float))
        np.save(outdir / "building_mask.npy", np.array(building_mask, dtype=float))
        np.save(outdir / "landcover.npy", np.array(landcover, dtype=float))
        np.save(outdir / "wall_height.npy", np.array(wall_height, dtype=float))
        np.save(outdir / "wall_aspect.npy", np.array(wall_aspect, dtype=float))

    metadata = {
        "created_from_blend": bpy.data.filepath,
        "grid": {
            "ncols": ncols,
            "nrows": nrows,
            "cellsize": cell,
            "local_xllcorner": x0_local,
            "local_yllcorner": y0_local,
            "geo_xllcorner": xllcorner_geo,
            "geo_yllcorner": yllcorner_geo,
            "local_extent": {
                "xmin": x0_local,
                "xmax": x1_local,
                "ymin": y0_local,
                "ymax": y1_local,
            },
            "epsg": args["epsg"],
        },
        "files": {
            "dsm": "dsm.asc",
            "dem": "dem.asc",
            "cdsm": "cdsm.asc",
            "tdsm": "tdsm.asc",
            "building_mask": "building_mask.asc",
            "landcover": "landcover.asc",
            "wall_height": "wall_height.asc",
            "wall_aspect": "wall_aspect.asc",
        },
        "settings": {
            "tree_class": args["tree_class"],
            "ground_class": args["ground_class"],
            "include_vehicles": args["include_vehicles"],
            "include_humans": args["include_humans"],
            "include_lamps": args["include_lamps"],
            "keep_ground_buffer": args["keep_ground_buffer"],
            "cdsm_min_height": args["cdsm_min_height"],
            "tdsm_min_height": args["tdsm_min_height"],
            "wall_min_height": args["wall_min_height"],
        },
        "derived": {
            "wall_height_aspect_method": "adjacent_nonbuilding_cells_from_dsm_dem_building_mask",
            **wall_stats,
        },
        "umep_landcover_codes": {
            "1": "paved",
            "2": "buildings",
            "3": "evergreen_trees",
            "4": "deciduous_trees",
            "5": "grass",
            "6": "bare_soil",
            "7": "water",
        },
    }
    (outdir / "solweig_grid_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _log("[SOLWEIG] Export complete.", args["verbose"])
    for fn in (
        "dsm.asc",
        "dem.asc",
        "cdsm.asc",
        "tdsm.asc",
        "building_mask.asc",
        "landcover.asc",
        "wall_height.asc",
        "wall_aspect.asc",
        "solweig_grid_meta.json",
    ):
        _log(f"[SOLWEIG] {outdir / fn}", args["verbose"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
