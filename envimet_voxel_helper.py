#!/usr/bin/env python3
"""Helper utilities for interactive ENVI-met voxel visualization.

This module provides a small reusable function that opens a 3D voxel plot using
matplotlib. In an interactive backend window, you can:
- rotate with left mouse drag
- pan with right mouse drag (backend-dependent)
- zoom with the scroll wheel

It is designed for arrays exported by blend_to_envimet_voxels.py, where 3D voxel
arrays use shape (nz, ny, nx).
"""

from __future__ import annotations

from pathlib import Path
import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEFAULT_COLORS = {
    "buildings_3d": "#d8b889",
    "canopy_3d": "#63c94f",
    "woody_3d": "#6f4e2a",
    "solid_3d": "#7f7f7f",
}

SOLID_LAYER_COLORS = {
    "building": DEFAULT_COLORS["buildings_3d"],
    "canopy": DEFAULT_COLORS["canopy_3d"],
    "woody": DEFAULT_COLORS["woody_3d"],
    "other": DEFAULT_COLORS["solid_3d"],
}


def _load_json_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return np.asarray(data)


def load_envimet_array(indir: str | Path, array_name: str) -> tuple[np.ndarray, dict]:
    """Load one ENVI-met voxel/grid array from an export folder.

    Parameters
    ----------
    indir:
        Folder containing envimet_voxel_meta.json and either envimet_voxels.npz,
        .npy files, or .json.gz files.
    array_name:
        One of surface_2d, dem_2d, top_2d, building_top_2d, canopy_top_2d,
        woody_top_2d, buildings_3d, canopy_3d, woody_3d, solid_3d.

    Returns
    -------
    (array, meta)
        array is a numpy ndarray, meta is the parsed metadata JSON.
    """
    indir = Path(indir)
    meta_path = indir / "envimet_voxel_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    npz_path = indir / "envimet_voxels.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        if array_name not in data.files:
            raise KeyError(f"{array_name!r} not found in {npz_path.name}. Available: {list(data.files)}")
        return data[array_name], meta

    npy_path = indir / f"envimet_{array_name}.npy"
    if npy_path.exists():
        return np.load(npy_path), meta

    json_gz_path = indir / f"envimet_{array_name}.json.gz"
    if json_gz_path.exists():
        return _load_json_gz(json_gz_path), meta

    raise FileNotFoundError(
        f"Could not find array {array_name!r} in {indir}. Checked envimet_voxels.npz, {npy_path.name}, and {json_gz_path.name}."
    )



def _infer_facecolors(mask_xyz: np.ndarray, color: str | np.ndarray | None = None) -> np.ndarray | str:
    if color is None:
        return "#7f7f7f"
    if isinstance(color, str):
        return color
    arr = np.asarray(color)
    if arr.shape == mask_xyz.shape:
        return arr
    raise ValueError("color array must have the same shape as the voxel mask")


def _downsample_color_array(color: np.ndarray, stride: tuple[int, int, int] | None) -> np.ndarray:
    if stride is None:
        return color
    sz, sy, sx = stride
    return color[::max(1, sz), ::max(1, sy), ::max(1, sx)]


def _solid_layer_facecolors(indir: str | Path, solid_zyx: np.ndarray) -> np.ndarray | None:
    try:
        buildings, _ = load_envimet_array(indir, "buildings_3d")
        canopy, _ = load_envimet_array(indir, "canopy_3d")
        woody, _ = load_envimet_array(indir, "woody_3d")
    except (FileNotFoundError, KeyError):
        return None

    if buildings.shape != solid_zyx.shape or canopy.shape != solid_zyx.shape or woody.shape != solid_zyx.shape:
        return None

    colors = np.full(solid_zyx.shape, SOLID_LAYER_COLORS["other"], dtype=object)
    colors[np.asarray(canopy).astype(bool)] = SOLID_LAYER_COLORS["canopy"]
    colors[np.asarray(woody).astype(bool)] = SOLID_LAYER_COLORS["woody"]
    colors[np.asarray(buildings).astype(bool)] = SOLID_LAYER_COLORS["building"]
    return colors



def visualize_voxels_interactive(
    voxels_zyx: np.ndarray,
    *,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    title: str = "Voxel view",
    color: str | np.ndarray | None = None,
    edgecolor: str = "k",
    linewidth: float = 0.05,
    alpha: float = 0.9,
    elev: float = 28.0,
    azim: float = -58.0,
    max_voxels: int = 120000,
    stride: tuple[int, int, int] | None = None,
    legend: dict[str, str] | None = None,
    hide_axes: bool = True,
    zoom_to_occupied: bool = True,
    zoom_padding: float = 1.5,
) -> tuple[plt.Figure, plt.Axes]:
    """Open an interactive 3D voxel window.

    Parameters
    ----------
    voxels_zyx:
        3D boolean or 0/1 occupancy array with shape (nz, ny, nx).
    voxel_size:
        Physical cell size as (dx, dy, dz).
    title:
        Figure title.
    color:
        Solid color string, or an array matching the voxel mask in z,y,x or x,y,z order.
    max_voxels:
        Safety cap for dense scenes. If occupied voxels exceed this and stride is
        not provided, the array is automatically downsampled.
    stride:
        Optional downsampling stride as (sz, sy, sx). Example: (1,2,2).
    hide_axes:
        Hide axis panes, ticks, and labels for a cleaner object-style view.
    zoom_to_occupied:
        Set the camera limits around occupied voxels instead of the whole grid.

    Returns
    -------
    (fig, ax)

    Notes
    -----
    In the opened matplotlib window, rotate and zoom with the mouse.
    """
    arr = np.asarray(voxels_zyx)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array with shape (nz, ny, nx), got {arr.shape}")

    mask = arr.astype(bool)
    occupied = int(mask.sum())
    if occupied == 0:
        raise ValueError("The voxel grid is empty; there is nothing to display.")

    color_arr = None
    color_order = None
    if color is not None and not isinstance(color, str):
        color_arr = np.asarray(color)
        if color_arr.shape != arr.shape:
            transposed_shape = (arr.shape[2], arr.shape[1], arr.shape[0])
            if color_arr.shape != transposed_shape:
                raise ValueError("color array must match the voxel mask in z,y,x or x,y,z order")
            color_order = "xyz"
        else:
            color_order = "zyx"

    if stride is None and occupied > max_voxels:
        factor = int(np.ceil((occupied / max_voxels) ** (1.0 / 3.0)))
        stride = (factor, factor, factor)
    if stride is not None:
        sz, sy, sx = stride
        mask = mask[::max(1, sz), ::max(1, sy), ::max(1, sx)]
        if color_arr is not None and color_order == "zyx":
            color_arr = _downsample_color_array(color_arr, stride)
        elif color_arr is not None and color_order == "xyz":
            color_arr = color_arr[::max(1, sx), ::max(1, sy), ::max(1, sz)]

    # Convert from (z, y, x) to (x, y, z) for matplotlib.voxels
    mask_xyz = np.transpose(mask, (2, 1, 0))
    if color_arr is not None:
        if color_order == "zyx":
            color = np.transpose(color_arr, (2, 1, 0))
        else:
            color = color_arr
    dx, dy, dz = voxel_size

    # Create grid corner coordinates so non-cubic voxels render with correct proportions
    nx, ny, nz = mask_xyz.shape
    x = np.arange(nx + 1, dtype=float) * dx
    y = np.arange(ny + 1, dtype=float) * dy
    z = np.arange(nz + 1, dtype=float) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    facecolors = _infer_facecolors(mask_xyz, color)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(
        X,
        Y,
        Z,
        mask_xyz,
        facecolors=facecolors,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.set_title(title)

    xlim = (0.0, max(dx * nx, 1e-6))
    ylim = (0.0, max(dy * ny, 1e-6))
    zlim = (0.0, max(dz * nz, 1e-6))
    if zoom_to_occupied:
        occupied_xyz = np.argwhere(mask_xyz)
        if occupied_xyz.size:
            lo = occupied_xyz.min(axis=0).astype(float)
            hi = occupied_xyz.max(axis=0).astype(float) + 1.0
            pad = max(0.0, float(zoom_padding))
            xlim = (max(0.0, (lo[0] - pad) * dx), min(dx * nx, (hi[0] + pad) * dx))
            ylim = (max(0.0, (lo[1] - pad) * dy), min(dy * ny, (hi[1] + pad) * dy))
            zlim = (max(0.0, (lo[2] - pad) * dz), min(dz * nz, (hi[2] + pad) * dz))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    if legend:
        handles = [Patch(facecolor=value, edgecolor=edgecolor, label=key) for key, value in legend.items()]
        ax.legend(handles=handles, loc="upper right")
    if hide_axes:
        ax.set_axis_off()
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((
        max(xlim[1] - xlim[0], 1e-6),
        max(ylim[1] - ylim[0], 1e-6),
        max(zlim[1] - zlim[0], 1e-6),
    ))
    plt.tight_layout()
    plt.show()
    return fig, ax



def visualize_envimet_array(
    indir: str | Path,
    array_name: str = "solid_3d",
    *,
    color: str | np.ndarray | None = None,
    max_voxels: int = 120000,
    stride: tuple[int, int, int] | None = None,
    hide_axes: bool = True,
    zoom_to_occupied: bool = True,
    zoom_padding: float = 1.5,
) -> tuple[plt.Figure, plt.Axes]:
    """Load one ENVI-met 3D array from disk and display it interactively."""
    arr, meta = load_envimet_array(indir, array_name)
    if arr.ndim != 3:
        raise ValueError(f"{array_name!r} is not a 3D array; got shape {arr.shape}")

    grid = meta.get("grid", {})
    dx = float(grid.get("dx", 1.0))
    dy = float(grid.get("dy", 1.0))
    dz = float(grid.get("dz", 1.0))
    legend = None
    if color is None:
        if array_name == "solid_3d":
            color = _solid_layer_facecolors(indir, arr)
            if color is not None:
                legend = {
                    "buildings": SOLID_LAYER_COLORS["building"],
                    "canopy": SOLID_LAYER_COLORS["canopy"],
                    "woody/trunks": SOLID_LAYER_COLORS["woody"],
                    "other solid": SOLID_LAYER_COLORS["other"],
                }
        if color is None:
            color = DEFAULT_COLORS.get(array_name, "#7f7f7f")

    return visualize_voxels_interactive(
        arr,
        voxel_size=(dx, dy, dz),
        title=array_name,
        color=color,
        max_voxels=max_voxels,
        stride=stride,
        legend=legend,
        hide_axes=hide_axes,
        zoom_to_occupied=zoom_to_occupied,
        zoom_padding=zoom_padding,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Interactive ENVI-met voxel viewer helper")
    ap.add_argument("--indir", required=True, help="Folder containing envimet_voxel_meta.json")
    ap.add_argument("--array", default="solid_3d", help="3D array name to view")
    ap.add_argument("--color", help="Optional matplotlib color")
    ap.add_argument("--max-voxels", type=int, default=120000)
    ap.add_argument("--stride", help="Optional stride as sz,sy,sx e.g. 1,2,2")
    ap.add_argument("--show-axes", action="store_true", help="Show axes, ticks, and labels")
    ap.add_argument("--no-zoom", action="store_true", help="Use full grid bounds instead of occupied voxel bounds")
    ap.add_argument("--zoom-padding", type=float, default=1.5, help="Padding around occupied voxels, in cells")
    ns = ap.parse_args()

    stride = None
    if ns.stride:
        parts = [int(p) for p in ns.stride.split(",")]
        if len(parts) != 3:
            raise SystemExit("--stride must have three integers: sz,sy,sx")
        stride = tuple(parts)

    visualize_envimet_array(
        ns.indir,
        array_name=ns.array,
        color=ns.color,
        max_voxels=ns.max_voxels,
        stride=stride,
        hide_axes=not ns.show_axes,
        zoom_to_occupied=not ns.no_zoom,
        zoom_padding=ns.zoom_padding,
    )
