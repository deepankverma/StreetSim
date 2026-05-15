#!/usr/bin/env python3
"""Export ENVI-met-style voxel grids from the current Blender scene."""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _parse_bool(text: str | bool | None, default: bool = False) -> bool:
    if text is None:
        return default
    if isinstance(text, bool):
        return text
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = {
        "outdir": None,
        "dx": 2.0,
        "dy": 2.0,
        "dz": 1.0,
        "padding": 2.0,
        "z_margin": 5.0,
        "origin_x": None,
        "origin_y": None,
        "epsg": None,
        "tree_class": "deciduous",
        "ground_class": "bare_soil",
        "include_vehicles": False,
        "include_humans": False,
        "include_lamps": False,
        "keep_ground_buffer": True,
        "canopy_min_height": 0.5,
        "woody_min_height": 0.5,
        "write_npz": True,
        "write_json_fallback": True,
        "verbose": True,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        key = a[2:].replace("-", "_") if a.startswith("--") else a
        if a == "--outdir" and i + 1 < len(argv):
            args["outdir"] = argv[i + 1]
            i += 2
        elif key in {"dx", "dy", "dz", "padding", "z_margin", "origin_x", "origin_y", "canopy_min_height", "woody_min_height"} and i + 1 < len(argv):
            args[key] = float(argv[i + 1])
            i += 2
        elif key == "epsg" and i + 1 < len(argv):
            args["epsg"] = int(argv[i + 1])
            i += 2
        elif key in {"tree_class", "ground_class"} and i + 1 < len(argv):
            args[key] = str(argv[i + 1]).strip().lower()
            i += 2
        elif key in {"include_vehicles", "include_humans", "include_lamps", "keep_ground_buffer", "write_npz", "write_json_fallback", "verbose"} and i + 1 < len(argv):
            args[key] = _parse_bool(argv[i + 1], bool(args[key]))
            i += 2
        else:
            i += 1
    if not args["outdir"]:
        raise SystemExit("Error: --outdir is required")
    args["dx"] = max(0.1, float(args["dx"]))
    args["dy"] = max(0.1, float(args["dy"]))
    args["dz"] = max(0.1, float(args["dz"]))
    args["padding"] = max(0.0, float(args["padding"]))
    return args


def dg():
    return bpy.context.evaluated_depsgraph_get()


def _name(obj) -> str:
    return obj.name.lower() if obj else ""


def _log(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg, flush=True)


def _world_bbox(obj) -> tuple[Vector, Vector]:
    eo = obj.evaluated_get(dg())
    mat = eo.matrix_world
    corners = [mat @ Vector(c) for c in eo.bound_box]
    mn = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    mx = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return mn, mx


def is_mesh(obj) -> bool:
    return bool(obj) and obj.type == "MESH" and not getattr(obj, "hide_render", False)


def is_vehicle(obj) -> bool:
    nm = _name(obj)
    return any(t in nm for t in ("parkedcar", "car_", "car.", "vehicle", "truck", "bus", "van", "bike_", "bicycle", "scooter"))


def is_human(obj) -> bool:
    return any(t in _name(obj) for t in ("human", "person", "pedestrian"))


def is_lamp(obj) -> bool:
    return any(t in _name(obj) for t in ("lamp", "lightpole", "streetlight"))


def is_roof(obj) -> bool:
    return "roof" in _name(obj)


def is_building(obj) -> bool:
    return "building" in _name(obj) or is_roof(obj)


def is_woody(obj) -> bool:
    nm = _name(obj)
    return any(t in nm for t in ("_wood", "trunk", "branch")) or (nm.startswith("tree_") and "leaves" not in nm)


def is_canopy(obj) -> bool:
    nm = _name(obj)
    return any(t in nm for t in ("leaf", "leaves", "canopy", "foliage")) or (nm.startswith("tree_") and "leaves" in nm)


def is_veg(obj) -> bool:
    nm = _name(obj)
    return is_canopy(obj) or is_woody(obj) or any(t in nm for t in ("shrub", "grass", "vegetation", "plant"))


def is_ground(obj, keep_ground_buffer: bool = True) -> bool:
    nm = _name(obj)
    tokens = ("driveway", "bikepath", "footpath", "median", "parking", "gutter", "ground", "plaza", "pavement", "sidewalk", "walkway", "road", "lane", "asphalt", "curb")
    if not any(t in nm for t in tokens):
        return False
    if not keep_ground_buffer and nm == "ground":
        return False
    return True


def ground_surface_code(obj, ground_class: str) -> int:
    nm = _name(obj)
    if any(t in nm for t in ("driveway", "bikepath", "footpath", "parking", "gutter", "road", "lane", "asphalt", "pavement", "curb")):
        return 1
    if "median" in nm:
        return 2
    if ground_class == "grass":
        return 2
    if ground_class == "paved":
        return 1
    return 3


class Hit:
    def __init__(self, ok: bool, location=None, obj=None):
        self.ok = ok
        self.location = location
        self.obj = obj


class SceneClassifier:
    def __init__(self, *, include_vehicles: bool, include_humans: bool, include_lamps: bool, keep_ground_buffer: bool):
        self.include_vehicles = include_vehicles
        self.include_humans = include_humans
        self.include_lamps = include_lamps
        self.keep_ground_buffer = keep_ground_buffer
        self.deps = dg()
        self.scene = bpy.context.scene

    def static_mesh(self, obj) -> bool:
        if not is_mesh(obj):
            return False
        if is_vehicle(obj) and not self.include_vehicles:
            return False
        if is_human(obj) and not self.include_humans:
            return False
        if is_lamp(obj) and not self.include_lamps:
            return False
        if _name(obj) == "ground" and not self.keep_ground_buffer:
            return False
        return True

    def extent_objects(self):
        out = []
        for obj in bpy.data.objects:
            if not self.static_mesh(obj):
                continue
            if is_ground(obj, self.keep_ground_buffer) or is_building(obj) or is_veg(obj):
                out.append(obj)
        return out

    def cast_down_until(self, origin: Vector, predicate, max_dist: float, max_hops: int = 32, eps: float = 1e-4) -> Hit:
        start = origin.copy()
        remain = float(max_dist)
        for _ in range(max_hops):
            hit, loc, _norm, _face_idx, obj, _mat = self.scene.ray_cast(self.deps, start, Vector((0, 0, -1)), distance=remain)
            if not hit:
                return Hit(False)
            if self.static_mesh(obj) and predicate(obj):
                return Hit(True, loc, obj)
            step = max((start.z - loc.z) + eps, eps)
            start = Vector((start.x, start.y, loc.z - eps))
            remain -= step
            if remain <= 0:
                break
        return Hit(False)


def _new_2d(rows: int, cols: int, value=0):
    return [[value for _ in range(cols)] for _ in range(rows)]


def _new_3d(layers: int, rows: int, cols: int):
    return [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(layers)]


def _fill_column(grid: list[list[list[int]]], row: int, col: int, z0: float, dz: float, bottom: float, top: float) -> int:
    if top <= bottom:
        return 0
    layers = len(grid)
    start = max(0, int(math.floor((bottom - z0) / dz)))
    end = min(layers - 1, int(math.floor((top - z0) / dz)))
    count = 0
    for k in range(start, end + 1):
        center = z0 + (k + 0.5) * dz
        if bottom <= center <= top:
            grid[k][row][col] = 1
            count += 1
    return count


def _write_json_gz(path: Path, value) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(value, f)


def _as_np(value, dtype):
    return np.asarray(value, dtype=dtype) if np is not None else None


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
        raise SystemExit("Error: no eligible meshes found for ENVI-met voxel export.")

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

    dx = float(args["dx"])
    dy = float(args["dy"])
    dz = float(args["dz"])
    padding = float(args["padding"])

    x0 = min_x - padding
    y0 = min_y - padding
    x1 = max_x + padding
    y1 = max_y + padding
    z0 = min(0.0, min_z)
    z1 = max_z + max(dz, float(args["z_margin"]))

    nx = max(1, int(math.ceil((x1 - x0) / dx)))
    ny = max(1, int(math.ceil((y1 - y0) / dy)))
    nz = max(1, int(math.ceil((z1 - z0) / dz)))
    x1 = x0 + nx * dx
    y1 = y0 + ny * dy
    z1 = z0 + nz * dz

    z_top = z1 + max(5.0, float(args["z_margin"]))
    max_dist = (z_top - min_z) + max(10.0, float(args["z_margin"]))

    _log(f"[ENVI-met] Grid: nx={nx} ny={ny} nz={nz} cell=({dx},{dy},{dz})", args["verbose"])

    surface_2d = _new_2d(ny, nx, 0)
    dem_2d = _new_2d(ny, nx, z0)
    top_2d = _new_2d(ny, nx, z0)
    building_top_2d = _new_2d(ny, nx, 0.0)
    canopy_top_2d = _new_2d(ny, nx, 0.0)
    woody_top_2d = _new_2d(ny, nx, 0.0)

    buildings_3d = _new_3d(nz, ny, nx)
    canopy_3d = _new_3d(nz, ny, nx)
    woody_3d = _new_3d(nz, ny, nx)

    counts = {"building_cells": 0, "canopy_cells": 0, "woody_cells": 0}

    for row in range(ny):
        y = y1 - (row + 0.5) * dy
        if args["verbose"] and row % max(1, ny // 10) == 0:
            _log(f"[ENVI-met] Row {row + 1}/{ny}", True)
        for col in range(nx):
            x = x0 + (col + 0.5) * dx
            origin = Vector((x, y, z_top))

            top_hit = classifier.cast_down_until(origin, lambda o: classifier.static_mesh(o), max_dist)
            ground_hit = classifier.cast_down_until(origin, lambda o: is_ground(o, classifier.keep_ground_buffer), max_dist)
            building_hit = classifier.cast_down_until(origin, is_building, max_dist)
            canopy_hit = classifier.cast_down_until(origin, is_canopy, max_dist)
            woody_hit = classifier.cast_down_until(origin, is_woody, max_dist)
            veg_hit = classifier.cast_down_until(origin, is_veg, max_dist)

            ground_z = float(ground_hit.location.z) if ground_hit.ok else z0
            dem_2d[row][col] = ground_z

            if top_hit.ok:
                top_2d[row][col] = float(top_hit.location.z)
            if ground_hit.ok:
                surface_2d[row][col] = ground_surface_code(ground_hit.obj, args["ground_class"])

            if building_hit.ok:
                building_top = float(building_hit.location.z)
                building_top_2d[row][col] = max(0.0, building_top - ground_z)
                counts["building_cells"] += _fill_column(buildings_3d, row, col, z0, dz, ground_z, building_top)

            if canopy_hit.ok:
                canopy_top = float(canopy_hit.location.z)
                h = max(0.0, canopy_top - ground_z)
                if h >= float(args["canopy_min_height"]):
                    canopy_top_2d[row][col] = h
                    counts["canopy_cells"] += _fill_column(canopy_3d, row, col, z0, dz, ground_z, canopy_top)
            elif veg_hit.ok:
                veg_top = float(veg_hit.location.z)
                h = max(0.0, veg_top - ground_z)
                if h >= float(args["canopy_min_height"]):
                    canopy_top_2d[row][col] = h
                    counts["canopy_cells"] += _fill_column(canopy_3d, row, col, z0, dz, ground_z, veg_top)

            if woody_hit.ok:
                woody_top = float(woody_hit.location.z)
                h = max(0.0, woody_top - ground_z)
                if h >= float(args["woody_min_height"]):
                    woody_top_2d[row][col] = h
                    counts["woody_cells"] += _fill_column(woody_3d, row, col, z0, dz, ground_z, woody_top)

    solid_3d = _new_3d(nz, ny, nx)
    for k in range(nz):
        for row in range(ny):
            for col in range(nx):
                solid_3d[k][row][col] = 1 if buildings_3d[k][row][col] or canopy_3d[k][row][col] or woody_3d[k][row][col] else 0

    arrays = {
        "surface_2d": surface_2d,
        "dem_2d": dem_2d,
        "top_2d": top_2d,
        "building_top_2d": building_top_2d,
        "canopy_top_2d": canopy_top_2d,
        "woody_top_2d": woody_top_2d,
        "buildings_3d": buildings_3d,
        "canopy_3d": canopy_3d,
        "woody_3d": woody_3d,
        "solid_3d": solid_3d,
    }

    files: dict[str, str | None] = {}
    if args["write_npz"] and np is not None:
        np.savez_compressed(
            outdir / "envimet_voxels.npz",
            surface_2d=_as_np(surface_2d, np.uint8),
            dem_2d=_as_np(dem_2d, np.float32),
            top_2d=_as_np(top_2d, np.float32),
            building_top_2d=_as_np(building_top_2d, np.float32),
            canopy_top_2d=_as_np(canopy_top_2d, np.float32),
            woody_top_2d=_as_np(woody_top_2d, np.float32),
            buildings_3d=_as_np(buildings_3d, np.uint8),
            canopy_3d=_as_np(canopy_3d, np.uint8),
            woody_3d=_as_np(woody_3d, np.uint8),
            solid_3d=_as_np(solid_3d, np.uint8),
        )
        files["npz"] = "envimet_voxels.npz"
    else:
        files["npz"] = None

    if args["write_json_fallback"]:
        for name, value in arrays.items():
            fn = f"{name}.json.gz"
            _write_json_gz(outdir / fn, value)
            files[name] = fn

    xll = float(args["origin_x"]) if args["origin_x"] is not None else x0
    yll = float(args["origin_y"]) if args["origin_y"] is not None else y0
    metadata = {
        "created_from_blend": bpy.data.filepath,
        "grid": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "local_xllcorner": x0,
            "local_yllcorner": y0,
            "local_zbottom": z0,
            "geo_xllcorner": xll,
            "geo_yllcorner": yll,
            "epsg": args["epsg"],
            "array_order_3d": "z,y,x",
            "array_order_2d": "y,x",
        },
        "surface_codes": {
            "0": "unknown",
            "1": "paved",
            "2": "grass_or_median",
            "3": "bare_soil",
        },
        "settings": {k: v for k, v in args.items() if k != "outdir"},
        "counts": counts,
        "files": files,
    }
    (outdir / "envimet_voxel_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _log(f"[ENVI-met] Export complete: {outdir}", args["verbose"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
