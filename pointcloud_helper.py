#!/usr/bin/env python3
"""Helper utilities for interactive semantic point-cloud visualization."""

from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_CLASS_LEGEND = {
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

DEFAULT_CLASS_RGB = {
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


def _load_meta(indir: Path) -> dict:
    meta_path = indir / "pointcloud_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _rgb_from_class_ids(class_id: np.ndarray) -> np.ndarray:
    out = np.empty((class_id.size, 3), dtype=np.float32)
    for cid, rgb in DEFAULT_CLASS_RGB.items():
        out[class_id == cid] = np.asarray(rgb, dtype=np.float32) / 255.0
    unknown = ~np.isin(class_id, list(DEFAULT_CLASS_RGB))
    if np.any(unknown):
        out[unknown] = np.asarray(DEFAULT_CLASS_RGB[0], dtype=np.float32) / 255.0
    return out


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    out = {
        "x": data["x"],
        "y": data["y"],
        "z": data["z"],
        "class_id": data["class_id"] if "class_id" in data.files else np.zeros_like(data["x"], dtype=np.uint8),
    }
    if "object_id" in data.files:
        out["object_id"] = data["object_id"]
    if "rgb" in data.files:
        rgb = np.asarray(data["rgb"], dtype=np.float32)
        if rgb.max(initial=0.0) > 1.0:
            rgb = rgb / 255.0
        out["rgb"] = rgb
    else:
        out["rgb"] = _rgb_from_class_ids(out["class_id"].astype(int))
    if "intensity" in data.files:
        out["intensity"] = data["intensity"]
    return out


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    if not rows:
        raise ValueError(f"No points found in {path}")

    def col(name: str, default: float = 0.0, dtype=float):
        return np.asarray([dtype(row.get(name) or default) for row in rows])

    class_id = col("class_id", dtype=int).astype(np.uint8)
    out = {
        "x": col("x"),
        "y": col("y"),
        "z": col("z"),
        "class_id": class_id,
    }
    if "object_id" in rows[0]:
        out["object_id"] = col("object_id", dtype=int).astype(np.int32)
    if {"red", "green", "blue"}.issubset(rows[0]):
        out["rgb"] = np.stack([col("red"), col("green"), col("blue")], axis=1).astype(np.float32) / 255.0
    else:
        out["rgb"] = _rgb_from_class_ids(class_id.astype(int))
    if "intensity" in rows[0]:
        out["intensity"] = col("intensity")
    return out


def _read_ply_ascii(path: Path) -> dict[str, np.ndarray]:
    properties: list[str] = []
    vertex_count = None
    header_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            header_lines += 1
            text = line.strip()
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property"):
                properties.append(text.split()[-1])
            elif text == "end_header":
                break

    if vertex_count is None:
        raise ValueError(f"Could not find vertex count in PLY header: {path}")
    if vertex_count == 0:
        raise ValueError(f"No points found in {path}")

    data = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count)
    if data.ndim == 1:
        data = data.reshape((1, -1))
    index = {name: i for i, name in enumerate(properties)}

    def values(name: str, default: float = 0.0):
        if name in index:
            return data[:, index[name]]
        return np.full(data.shape[0], default, dtype=float)

    class_name = "classification" if "classification" in index else "class_id"
    class_id = values(class_name).astype(np.uint8)
    out = {
        "x": values("x"),
        "y": values("y"),
        "z": values("z"),
        "class_id": class_id,
    }
    if "object_id" in index:
        out["object_id"] = values("object_id").astype(np.int32)
    if {"red", "green", "blue"}.issubset(index):
        out["rgb"] = np.stack([values("red"), values("green"), values("blue")], axis=1).astype(np.float32) / 255.0
    else:
        out["rgb"] = _rgb_from_class_ids(class_id.astype(int))
    if "intensity" in index:
        out["intensity"] = values("intensity")
    return out


def load_pointcloud(indir: str | Path, file: str | Path | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Load an exported point cloud from NPZ, PLY, or CSV."""
    indir = Path(indir)
    meta = _load_meta(indir)
    if file is not None:
        path = Path(file)
        if not path.is_absolute():
            path = indir / path
    else:
        candidates = [
            indir / "airborne_pointcloud.npz",
            indir / "airborne_pointcloud.ply",
            indir / "airborne_pointcloud.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            names = ", ".join(p.name for p in candidates)
            raise FileNotFoundError(f"Could not find a point cloud in {indir}. Checked: {names}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _read_npz(path), meta
    if suffix == ".ply":
        return _read_ply_ascii(path), meta
    if suffix == ".csv":
        return _read_csv(path), meta
    raise ValueError(f"Unsupported point-cloud file type: {path}")


def _class_legend(meta: dict, class_ids: np.ndarray) -> dict[int, str]:
    raw = meta.get("class_legend") or {}
    legend = {int(k): str(v) for k, v in raw.items()} if raw else dict(DEFAULT_CLASS_LEGEND)
    return {cid: legend.get(cid, f"class {cid}") for cid in sorted(int(v) for v in np.unique(class_ids))}


def _name_class_id(name: str) -> int | None:
    nm = name.lower()
    if not nm:
        return None
    if any(t in nm for t in ("water", "pond", "fountain")):
        return 9
    if any(t in nm for t in ("lamp", "lightpole", "streetlight")):
        return 8
    if any(t in nm for t in ("human", "person", "pedestrian")):
        return 7
    if any(t in nm for t in ("parkedcar", "car_", "car.", "vehicle", "truck", "bus", "van", "bike_", "bicycle", "scooter")):
        return 6
    if any(t in nm for t in ("leaf", "leaves", "canopy", "foliage")):
        return 4
    if any(t in nm for t in ("_wood", "trunk", "branch", "stem", "bark")) or nm.startswith("tree_"):
        return 5
    if "roof" in nm:
        return 3
    if "building" in nm:
        return 2
    if any(t in nm for t in ("driveway", "bikepath", "footpath", "median", "parking", "gutter", "ground", "plaza", "pavement", "sidewalk", "walkway", "road", "lane", "asphalt", "curb")):
        return 1
    return None


def _repair_class_ids_from_object_names(points: dict[str, np.ndarray], meta: dict) -> dict[str, np.ndarray]:
    if "object_id" not in points:
        return points
    raw_names = meta.get("object_names") or {}
    if not raw_names:
        return points

    object_names = {int(k): str(v) for k, v in raw_names.items()}
    object_id = np.asarray(points["object_id"], dtype=int)
    class_id = np.asarray(points.get("class_id", np.zeros_like(object_id)), dtype=np.uint8).copy()
    changed = False
    for oid in np.unique(object_id):
        inferred = _name_class_id(object_names.get(int(oid), ""))
        if inferred is None:
            continue
        mask = object_id == int(oid)
        if np.any(class_id[mask] != inferred):
            class_id[mask] = inferred
            changed = True
    if changed:
        points = dict(points)
        points["class_id"] = class_id
    return points


def _filter_classes(points: dict[str, np.ndarray], meta: dict, classes: str | None) -> dict[str, np.ndarray]:
    if not classes:
        return points

    legend = _class_legend(meta, points["class_id"])
    name_to_id = {name.lower(): cid for cid, name in legend.items()}
    wanted: set[int] = set()
    for token in classes.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.isdigit():
            wanted.add(int(token))
        elif token in name_to_id:
            wanted.add(name_to_id[token])
        else:
            raise ValueError(f"Unknown class filter {token!r}. Available: {', '.join(legend.values())}")

    mask = np.isin(points["class_id"].astype(int), list(wanted))
    if not np.any(mask):
        raise ValueError("Class filter removed all points; nothing to display.")
    return {key: value[mask] if value.shape[0] == mask.shape[0] else value for key, value in points.items()}


def _sample_points(points: dict[str, np.ndarray], max_points: int, seed: int) -> dict[str, np.ndarray]:
    n = int(points["x"].shape[0])
    if n <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return {key: value[idx] if value.shape[0] == n else value for key, value in points.items()}


def visualize_pointcloud_interactive(
    points: dict[str, np.ndarray],
    meta: dict | None = None,
    *,
    title: str = "Point-cloud view",
    color_by: str = "class",
    point_size: float = 3.0,
    alpha: float = 0.85,
    max_points: int = 250000,
    sample_seed: int = 42,
    elev: float = 28.0,
    azim: float = -58.0,
    hide_axes: bool = True,
    zoom_to_points: bool = True,
    zoom_padding_frac: float = 0.005,
    zoom_factor: float = 0.72,
) -> tuple[plt.Figure, plt.Axes]:
    """Open an interactive 3D point-cloud window."""
    meta = meta or {}
    points = _sample_points(points, max_points=max_points, seed=sample_seed)
    x = np.asarray(points["x"], dtype=float)
    y = np.asarray(points["y"], dtype=float)
    z = np.asarray(points["z"], dtype=float)
    if x.size == 0:
        raise ValueError("The point cloud is empty; there is nothing to display.")

    class_id = np.asarray(points.get("class_id", np.zeros_like(x)), dtype=int)
    if color_by == "class":
        colors = _rgb_from_class_ids(class_id)
    elif color_by == "height":
        colors = z
    elif color_by == "intensity":
        if "intensity" not in points:
            raise ValueError("This point cloud has no intensity values.")
        colors = points["intensity"]
    else:
        raise ValueError("--color-by must be one of: class, height, intensity")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(x, y, z, c=colors, s=point_size, alpha=alpha, linewidths=0, depthshade=False)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    if zoom_to_points:
        x_center = float((x.min() + x.max()) * 0.5)
        y_center = float((y.min() + y.max()) * 0.5)
        z_center = float((z.min() + z.max()) * 0.5)
        zoom = min(1.0, max(0.05, float(zoom_factor)))
        x_span = max(float(x.max() - x.min()) * zoom, 1e-6)
        y_span = max(float(y.max() - y.min()) * zoom, 1e-6)
        z_span = max(float(z.max() - z.min()) * zoom, 1e-6)
        pad = max(x_span, y_span, z_span) * max(0.0, float(zoom_padding_frac))
        xlim = (x_center - x_span * 0.5 - pad, x_center + x_span * 0.5 + pad)
        ylim = (y_center - y_span * 0.5 - pad, y_center + y_span * 0.5 + pad)
        zlim = (z_center - z_span * 0.5 - pad, z_center + z_span * 0.5 + pad)
    else:
        xlim = (float(x.min()), float(x.max()))
        ylim = (float(y.min()), float(y.max()))
        zlim = (float(z.min()), float(z.max()))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((
        max(xlim[1] - xlim[0], 1e-6),
        max(ylim[1] - ylim[0], 1e-6),
        max(zlim[1] - zlim[0], 1e-6),
    ))

    if color_by == "class":
        legend = _class_legend(meta, class_id)
        handles = []
        for cid, label in legend.items():
            rgb = np.asarray(DEFAULT_CLASS_RGB.get(cid, DEFAULT_CLASS_RGB[0]), dtype=float) / 255.0
            handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=rgb, markersize=7, label=label))
        ax.legend(handles=handles, loc="upper right", markerscale=1.5)
    else:
        fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02, label=color_by)

    if hide_axes:
        ax.set_axis_off()
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()
    return fig, ax


def visualize_pointcloud_export(
    indir: str | Path,
    *,
    file: str | Path | None = None,
    classes: str | None = None,
    color_by: str = "class",
    point_size: float = 3.0,
    alpha: float = 0.85,
    max_points: int = 250000,
    sample_seed: int = 42,
    hide_axes: bool = True,
    zoom_to_points: bool = True,
    zoom_factor: float = 0.72,
) -> tuple[plt.Figure, plt.Axes]:
    points, meta = load_pointcloud(indir, file=file)
    points = _repair_class_ids_from_object_names(points, meta)
    points = _filter_classes(points, meta, classes)
    return visualize_pointcloud_interactive(
        points,
        meta,
        title="airborne_pointcloud",
        color_by=color_by,
        point_size=point_size,
        alpha=alpha,
        max_points=max_points,
        sample_seed=sample_seed,
        hide_axes=hide_axes,
        zoom_to_points=zoom_to_points,
        zoom_factor=zoom_factor,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Interactive semantic point-cloud viewer helper")
    ap.add_argument("--indir", required=True, help="Folder containing pointcloud_meta.json and airborne_pointcloud.*")
    ap.add_argument("--file", help="Optional point-cloud file name/path. Default: NPZ, then PLY, then CSV.")
    ap.add_argument("--classes", help="Optional comma-separated class ids/names, e.g. building,canopy,woody")
    ap.add_argument("--color-by", default="class", choices=["class", "height", "intensity"])
    ap.add_argument("--point-size", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--max-points", type=int, default=250000)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--zoom-factor", type=float, default=0.72, help="Fraction of full point extent to show; smaller is more zoomed in")
    ap.add_argument("--show-axes", action="store_true", help="Show axes, ticks, and labels")
    ap.add_argument("--no-zoom", action="store_true", help="Use exact point bounds without visual padding")
    ns = ap.parse_args()

    visualize_pointcloud_export(
        ns.indir,
        file=ns.file,
        classes=ns.classes,
        color_by=ns.color_by,
        point_size=ns.point_size,
        alpha=ns.alpha,
        max_points=ns.max_points,
        sample_seed=ns.sample_seed,
        hide_axes=not ns.show_axes,
        zoom_to_points=not ns.no_zoom,
        zoom_factor=ns.zoom_factor,
    )
