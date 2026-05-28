#!/usr/bin/env python3
"""Export an airborne-style semantic point cloud from a Blender scene."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


CLASS_LEGEND = {
    0: "other",
    1: "ground",
    2: "building",
    3: "roof",
    4: "canopy",
    5: "woody",
    6: "vehicle",
    7: "human",
    8: "lamp",
    9: "water",
}

CLASS_RGB = {
    0: (255, 0, 180),
    1: (235, 235, 225),
    2: (216, 184, 137),
    3: (196, 156, 108),
    4: (99, 201, 79),
    5: (111, 78, 42),
    6: (70, 110, 220),
    7: (230, 200, 60),
    8: (180, 120, 220),
    9: (70, 180, 230),
}


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
        "spacing": 0.5,
        "padding": 2.0,
        "z_margin": 80.0,
        "origin_x": None,
        "origin_y": None,
        "epsg": None,
        "scan_count": 3,
        "jitter_frac": 0.35,
        "noise_xy": 0.02,
        "noise_z": 0.03,
        "dropout": 0.0,
        "ground_return_prob": 0.25,
        "include_vehicles": True,
        "include_humans": True,
        "include_lamps": False,
        "keep_ground_buffer": True,
        "write_ply": True,
        "write_npz": True,
        "write_csv": False,
        "seed": 42,
        "verbose": True,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--outdir" and i + 1 < len(argv):
            args["outdir"] = argv[i + 1]
            i += 2
        elif a in {"--spacing", "--padding", "--z-margin", "--jitter-frac", "--noise-xy", "--noise-z", "--dropout", "--ground-return-prob"} and i + 1 < len(argv):
            args[a[2:].replace("-", "_")] = float(argv[i + 1])
            i += 2
        elif a in {"--origin-x", "--origin-y"} and i + 1 < len(argv):
            args[a[2:].replace("-", "_")] = float(argv[i + 1])
            i += 2
        elif a == "--epsg" and i + 1 < len(argv):
            args["epsg"] = int(argv[i + 1])
            i += 2
        elif a in {"--scan-count", "--seed"} and i + 1 < len(argv):
            args[a[2:].replace("-", "_")] = int(argv[i + 1])
            i += 2
        elif a in {"--include-vehicles", "--include-humans", "--include-lamps", "--keep-ground-buffer", "--write-ply", "--write-npz", "--write-csv", "--verbose"} and i + 1 < len(argv):
            args[a[2:].replace("-", "_")] = _parse_bool(argv[i + 1], bool(args[a[2:].replace("-", "_")]))
            i += 2
        else:
            i += 1
    if not args["outdir"]:
        raise SystemExit("Error: --outdir is required")
    args["scan_count"] = max(1, int(args["scan_count"]))
    args["spacing"] = max(0.05, float(args["spacing"]))
    args["padding"] = max(0.0, float(args["padding"]))
    args["dropout"] = min(1.0, max(0.0, float(args["dropout"])))
    args["ground_return_prob"] = min(1.0, max(0.0, float(args["ground_return_prob"])))
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
    return "building" in _name(obj) and not is_roof(obj)


def is_canopy(obj) -> bool:
    nm = _name(obj)
    return any(t in nm for t in ("leaf", "leaves", "canopy", "foliage"))


def is_woody(obj) -> bool:
    nm = _name(obj)
    if is_canopy(obj):
        return False
    return any(t in nm for t in ("_wood", "trunk", "branch", "stem", "bark")) or nm.startswith("tree_")


def is_water(obj) -> bool:
    return any(t in _name(obj) for t in ("water", "pond", "fountain"))


def is_ground(obj, keep_ground_buffer: bool = True) -> bool:
    nm = _name(obj)
    tokens = ("driveway", "bikepath", "footpath", "median", "parking", "gutter", "ground", "plaza", "pavement", "sidewalk", "walkway", "road", "lane", "asphalt", "curb")
    if not any(t in nm for t in tokens):
        return False
    if not keep_ground_buffer and nm == "ground":
        return False
    return True


class SceneClassifier:
    def __init__(self, *, include_vehicles: bool, include_humans: bool, include_lamps: bool, keep_ground_buffer: bool):
        self.include_vehicles = include_vehicles
        self.include_humans = include_humans
        self.include_lamps = include_lamps
        self.keep_ground_buffer = keep_ground_buffer
        self.deps = dg()
        self.scene = bpy.context.scene

    def eligible(self, obj) -> bool:
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
        return [obj for obj in bpy.data.objects if self.eligible(obj)]

    def class_id(self, obj) -> int:
        if obj is None:
            return 0
        if is_vehicle(obj):
            return 6
        if is_human(obj):
            return 7
        if is_lamp(obj):
            return 8
        if is_water(obj):
            return 9
        if is_woody(obj):
            return 5
        if is_canopy(obj):
            return 4
        if is_roof(obj):
            return 3
        if is_building(obj):
            return 2
        if is_ground(obj, self.keep_ground_buffer):
            return 1
        return 0

    def is_vegetation(self, obj) -> bool:
        return self.class_id(obj) in {4, 5}

    def ray_cast_filtered(self, origin: Vector, direction: Vector, distance: float, *, skip_pred=None, max_hops: int = 32, eps: float = 1e-4):
        start = origin.copy()
        remain = float(distance)
        for _ in range(max_hops):
            hit, loc, norm, _face_idx, obj, _mat = self.scene.ray_cast(self.deps, start, direction, distance=remain)
            if not hit:
                return False, None, None, None
            should_skip = (not self.eligible(obj)) or (skip_pred(obj) if skip_pred else False)
            if not should_skip:
                return True, loc, norm, obj
            step = (loc - start).length + eps
            start = start + direction * step
            remain = max(0.0, remain - step)
        return False, None, None, None


def _intensity_from_normal(normal: Vector | None) -> int:
    if normal is None:
        return 128
    d = max(0.0, min(1.0, abs(normal.normalized().dot(Vector((0, 0, 1))))))
    return int(round(40 + d * 215))


def _write_ply_ascii(path: Path, points: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("comment generated by blend_to_pointcloud.py\n")
        f.write(f"element vertex {len(points)}\n")
        for prop in (
            "property float x", "property float y", "property float z",
            "property float nx", "property float ny", "property float nz",
            "property uchar red", "property uchar green", "property uchar blue",
            "property uchar class_id", "property int object_id",
            "property uchar return_number", "property ushort scan_id",
            "property uchar intensity",
            "property float classification",
            "property float class_id_sf",
            "property float object_id_sf",
            "property float return_number_sf",
            "property float scan_id_sf",
        ):
            f.write(prop + "\n")
        f.write("end_header\n")
        for p in points:
            f.write(
                f"{p['x']:.6f} {p['y']:.6f} {p['z']:.6f} {p['nx']:.6f} {p['ny']:.6f} {p['nz']:.6f} "
                f"{p['red']} {p['green']} {p['blue']} {p['class_id']} {p['object_id']} {p['return_number']} {p['scan_id']} {p['intensity']} "
                f"{float(p['class_id']):.1f} {float(p['class_id']):.1f} {float(p['object_id']):.1f} "
                f"{float(p['return_number']):.1f} {float(p['scan_id']):.1f}\n"
            )


def _write_csv(path: Path, points: list[dict], object_names: dict[int, str]) -> None:
    fields = ["x", "y", "z", "nx", "ny", "nz", "class_id", "class_name", "object_id", "object_name", "return_number", "scan_id", "intensity", "red", "green", "blue"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in points:
            row = dict(p)
            row["class_name"] = CLASS_LEGEND.get(int(p["class_id"]), "other")
            row["object_name"] = object_names.get(int(p["object_id"]), "")
            writer.writerow(row)


def _write_npz(path: Path, points: list[dict]) -> bool:
    if np is None:
        return False
    np.savez_compressed(
        path,
        x=np.asarray([p["x"] for p in points], dtype=np.float32),
        y=np.asarray([p["y"] for p in points], dtype=np.float32),
        z=np.asarray([p["z"] for p in points], dtype=np.float32),
        nx=np.asarray([p["nx"] for p in points], dtype=np.float32),
        ny=np.asarray([p["ny"] for p in points], dtype=np.float32),
        nz=np.asarray([p["nz"] for p in points], dtype=np.float32),
        class_id=np.asarray([p["class_id"] for p in points], dtype=np.uint8),
        object_id=np.asarray([p["object_id"] for p in points], dtype=np.int32),
        return_number=np.asarray([p["return_number"] for p in points], dtype=np.uint8),
        scan_id=np.asarray([p["scan_id"] for p in points], dtype=np.uint16),
        intensity=np.asarray([p["intensity"] for p in points], dtype=np.uint8),
        rgb=np.asarray([[p["red"], p["green"], p["blue"]] for p in points], dtype=np.uint8),
    )
    return True


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
        raise SystemExit("Error: no eligible meshes found for point-cloud export.")

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

    spacing = args["spacing"]
    padding = args["padding"]
    x0 = min_x - padding
    y0 = min_y - padding
    x1 = max_x + padding
    y1 = max_y + padding
    nx = max(1, int(math.ceil((x1 - x0) / spacing)))
    ny = max(1, int(math.ceil((y1 - y0) / spacing)))
    x1 = x0 + nx * spacing
    y1 = y0 + ny * spacing
    z_top = max_z + max(10.0, float(args["z_margin"]))
    max_dist = (z_top - min_z) + max(20.0, float(args["z_margin"]))

    _log(f"[PointCloud] grid={nx}x{ny} spacing={spacing:.3f}", args["verbose"])

    rng = random.Random(int(args["seed"]))
    jitter = float(args["jitter_frac"]) * spacing
    points: list[dict] = []
    object_name_to_id: dict[str, int] = {"": 0}
    class_counts: dict[str, int] = {}

    def object_id_for(obj) -> int:
        name = obj.name if obj else ""
        if name not in object_name_to_id:
            object_name_to_id[name] = len(object_name_to_id)
        return object_name_to_id[name]

    def add_point(loc: Vector, norm: Vector | None, obj, *, scan_id: int, return_number: int) -> None:
        cid = classifier.class_id(obj)
        rgb = CLASS_RGB.get(cid, CLASS_RGB[0])
        p = {
            "x": float(loc.x + (rng.gauss(0.0, args["noise_xy"]) if args["noise_xy"] > 0 else 0.0)),
            "y": float(loc.y + (rng.gauss(0.0, args["noise_xy"]) if args["noise_xy"] > 0 else 0.0)),
            "z": float(loc.z + (rng.gauss(0.0, args["noise_z"]) if args["noise_z"] > 0 else 0.0)),
            "nx": float(norm.x if norm is not None else 0.0),
            "ny": float(norm.y if norm is not None else 0.0),
            "nz": float(norm.z if norm is not None else 1.0),
            "red": int(rgb[0]),
            "green": int(rgb[1]),
            "blue": int(rgb[2]),
            "class_id": int(cid),
            "object_id": int(object_id_for(obj)),
            "return_number": int(return_number),
            "scan_id": int(scan_id),
            "intensity": _intensity_from_normal(norm),
        }
        points.append(p)
        label = CLASS_LEGEND.get(cid, "other")
        class_counts[label] = class_counts.get(label, 0) + 1

    direction = Vector((0.0, 0.0, -1.0))
    for scan_idx in range(args["scan_count"]):
        x_off = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        y_off = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        _log(f"[PointCloud] scan {scan_idx + 1}/{args['scan_count']}", args["verbose"])
        for row in range(ny):
            y = y0 + (row + 0.5) * spacing + y_off
            if y < y0 or y > y1:
                continue
            for col in range(nx):
                if args["dropout"] > 0.0 and rng.random() < args["dropout"]:
                    continue
                x = x0 + (col + 0.5) * spacing + x_off
                if x < x0 or x > x1:
                    continue
                origin = Vector((x, y, z_top))
                hit, loc, norm, obj = classifier.ray_cast_filtered(origin, direction, max_dist)
                if not hit:
                    continue
                add_point(loc, norm, obj, scan_id=scan_idx, return_number=1)
                if classifier.is_vegetation(obj) and args["ground_return_prob"] > 0.0 and rng.random() < args["ground_return_prob"]:
                    start = loc + direction * 0.02
                    remaining = max(0.1, max_dist - (origin - start).length)
                    hit2, loc2, norm2, obj2 = classifier.ray_cast_filtered(start, direction, remaining, skip_pred=classifier.is_vegetation)
                    if hit2:
                        add_point(loc2, norm2, obj2, scan_id=scan_idx, return_number=2)

    object_names = {idx: name for name, idx in object_name_to_id.items()}
    files: dict[str, str | None] = {}
    if args["write_ply"]:
        _write_ply_ascii(outdir / "airborne_pointcloud.ply", points)
        files["ply"] = "airborne_pointcloud.ply"
    else:
        files["ply"] = None
    if args["write_npz"]:
        files["npz"] = "airborne_pointcloud.npz" if _write_npz(outdir / "airborne_pointcloud.npz", points) else None
    else:
        files["npz"] = None
    if args["write_csv"]:
        _write_csv(outdir / "airborne_pointcloud.csv", points, object_names)
        files["csv"] = "airborne_pointcloud.csv"
    else:
        files["csv"] = None

    xll = float(args["origin_x"]) if args["origin_x"] is not None else x0
    yll = float(args["origin_y"]) if args["origin_y"] is not None else y0
    metadata = {
        "created_from_blend": bpy.data.filepath,
        "point_count": len(points),
        "class_counts": class_counts,
        "class_legend": {str(k): v for k, v in CLASS_LEGEND.items()},
        "object_names": {str(k): v for k, v in object_names.items()},
        "grid": {
            "nx": nx,
            "ny": ny,
            "spacing": spacing,
            "local_xllcorner": x0,
            "local_yllcorner": y0,
            "geo_xllcorner": xll,
            "geo_yllcorner": yll,
            "epsg": args["epsg"],
        },
        "settings": {k: v for k, v in args.items() if k != "outdir"},
        "files": files,
    }
    (outdir / "pointcloud_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _log(f"[PointCloud] wrote {len(points)} points to {outdir}", args["verbose"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
