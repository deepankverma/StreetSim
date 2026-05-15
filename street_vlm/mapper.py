"""Mapping from vision schema JSON to street_data.json."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .defaults import DEFAULT_POLICY


def _round_step(value: float, step: float = 0.5) -> float:
    if step <= 0:
        return float(value)
    return round(float(value) / step) * step


def _clamp_nonneg(value: float) -> float:
    return max(0.0, float(value or 0.0))


def _bool_present(item: dict, min_conf: float) -> bool:
    return bool(item.get("present", False)) and float(item.get("confidence", 0.0)) >= min_conf


def _count_from_visible(visible: int, present: bool, category: str, policy: dict) -> int:
    visible = max(0, int(visible or 0))
    if not present:
        return 0

    mode = policy.get("visible_to_total_mode", "visible_only")
    multiplier = float(policy.get("count_multiplier", {}).get(category, 1.0))
    minimum = int(policy.get("minimum_counts_when_present", {}).get(category, 1))
    max_buildings = int(policy.get("max_buildings_per_side", 999999))
    max_trees = int(policy.get("max_trees_per_side", 999999))

    if mode == "multiplier":
        out = int(round(visible * multiplier))
        result = max(minimum, out)
    elif mode == "clamp_minimum":
        result = max(minimum, visible)
    else:
        result = visible

    if category == "buildings":
        result = min(result, max_buildings)
    if category in {"trees", "median_trees"}:
        result = min(result, max_trees)
    return result


def _density_from_score(score: int, divisor: float) -> float:
    score = max(0, min(100, int(score or 0)))
    return score / float(divisor)


def _side_to_config(side_v: dict, policy: dict) -> dict:
    min_conf = float(policy.get("min_confidence_keep", 0.45))
    step = float(policy.get("round_width_to", 0.5))

    driveway_present = _bool_present(side_v["driveway"], min_conf)
    bike_present = _bool_present(side_v["bikepath"], min_conf)
    foot_present = _bool_present(side_v["footpath"], min_conf)
    buildings_present = bool(side_v["buildings"].get("present", False))
    trees_present = bool(side_v["trees"].get("present", False))

    driveway_width = _round_step(_clamp_nonneg(side_v["driveway"].get("width_m", 0.0)), step) if driveway_present else 0.0
    bike_width = _round_step(_clamp_nonneg(side_v["bikepath"].get("width_m", 0.0)), step) if bike_present else 0.0
    foot_width = _round_step(_clamp_nonneg(side_v["footpath"].get("width_m", 0.0)), step) if foot_present else 0.0

    building_count = _count_from_visible(
        side_v["buildings"].get("count_visible", 0),
        buildings_present,
        "buildings",
        policy,
    )
    tree_count = _count_from_visible(
        side_v["trees"].get("count_visible", 0),
        trees_present,
        "trees",
        policy,
    )

    return {
        "driveway": {"present": driveway_present and driveway_width > 0, "width": driveway_width},
        "bikepath": {"present": bike_present and bike_width > 0, "width": bike_width},
        "footpath": {"present": foot_present and foot_width > 0, "width": foot_width},
        "buildings": {"present": buildings_present and building_count > 0, "count": building_count},
        "trees": {"present": trees_present and tree_count > 0, "count": tree_count},
    }


def _parking_to_config(vision: dict, policy: dict) -> tuple[dict, dict]:
    min_conf = float(policy.get("min_confidence_keep", 0.45))
    step = float(policy.get("round_width_to", 0.5))

    l_parking_present = _bool_present(vision["left_side"]["parking"], min_conf)
    r_parking_present = _bool_present(vision["right_side"]["parking"], min_conf)

    l_width = _round_step(_clamp_nonneg(vision["left_side"]["parking"].get("bay_width_m", 0.0)), step) if l_parking_present else 0.0
    r_width = _round_step(_clamp_nonneg(vision["right_side"]["parking"].get("bay_width_m", 0.0)), step) if r_parking_present else 0.0

    park_present = l_parking_present or r_parking_present
    if l_parking_present and r_parking_present:
        sides = "both"
    elif l_parking_present:
        sides = "left"
    elif r_parking_present:
        sides = "right"
    else:
        sides = "none"

    street_parking = (
        {
            "present": True,
            "sides": sides,
            "width_m_per_side": {"left": l_width, "right": r_width},
            "min_driveway_width": float(policy.get("parking_min_driveway_width", 3.0)),
        }
        if park_present
        else {"present": False}
    )

    parked_count_visible = int(vision.get("dynamic", {}).get("parked_cars_visible", 0) or 0)
    parked_count = _count_from_visible(parked_count_visible, park_present and parked_count_visible > 0, "parked_cars", policy)
    parked_cars = {"present": parked_count > 0, "count": parked_count, "seed": 0} if parked_count > 0 else {"present": False, "count": 0, "seed": 0}

    return street_parking, parked_cars


def _median_to_config(vision: dict, policy: dict) -> dict:
    min_conf = float(policy.get("min_confidence_keep", 0.45))
    step = float(policy.get("round_width_to", 0.5))
    median_v = vision["median"]

    present = bool(median_v.get("present", False)) and float(median_v.get("confidence", 0.0)) >= min_conf
    width = _round_step(_clamp_nonneg(median_v.get("width_m", 0.0)), step) if present else 0.0
    trees_present = bool(median_v.get("trees_present", False)) and present
    tree_count = _count_from_visible(median_v.get("tree_count_visible", 0), trees_present, "median_trees", policy)

    return {
        "present": present and width > 0,
        "width": width,
        "trees": {"present": trees_present and tree_count > 0, "count": tree_count},
    }


def vision_to_street_data(vision: Dict[str, Any], policy: Dict[str, Any] | None = None) -> Dict[str, Any]:
    policy = deepcopy(DEFAULT_POLICY if policy is None else policy)

    left = _side_to_config(vision["left_side"], policy)
    right = _side_to_config(vision["right_side"], policy)
    median = _median_to_config(vision, policy)
    street_parking, parked_cars = _parking_to_config(vision, policy)

    any_drive = bool(left["driveway"]["present"] or right["driveway"]["present"])

    moving_visible = int(vision.get("dynamic", {}).get("moving_cars_visible", 0) or 0)
    moving_count = _count_from_visible(moving_visible, any_drive and moving_visible > 0, "moving_cars", policy)
    cars = (
        {"present": True, "count": moving_count, "scale": float(policy.get("cars_scale", 10.0))}
        if any_drive and moving_count > 0
        else {"present": False, "count": 0, "scale": float(policy.get("cars_scale", 10.0))}
    )

    ped_visible = int(vision.get("dynamic", {}).get("pedestrians_visible", 0) or 0)
    ped_count = _count_from_visible(ped_visible, ped_visible > 0, "pedestrians", policy)
    ped_count = max(int(policy.get("minimum_humans_total", 1)), ped_count)                 ## can remove later
    humans = (
        {
            "present": True,
            "count": ped_count,
            "scale": float(policy.get("humans_scale", 5.0)),
            "z_offset": float(policy.get("humans_z_offset", 1.2)),
        }
        if ped_count > 0
        else {
            "present": False,
            "count": 0,
            "scale": float(policy.get("humans_scale", 5.0)),
            "z_offset": float(policy.get("humans_z_offset", 1.2)),
        }
    )

    veg_present = bool(vision.get("vegetation", {}).get("present", False))
    veg_density = _density_from_score(
        vision.get("vegetation", {}).get("density_score_0_100", 0),
        float(policy.get("vegetation_density_divisor", 100.0)),
    )
    vegetation = {"present": True, "density": veg_density} if veg_present else {"present": False, "density": 0.0}

    length = float(vision.get("street_length_m_estimate") or policy.get("default_length_m", 50.0))
    if length <= 0:
        length = float(policy.get("default_length_m", 50.0))

    return {
        "schemaVersion": 3,
        "length": length,
        "sides": {"left": left, "right": right},
        "median": median,
        "street_parking": street_parking,
        "parked_cars": parked_cars,
        "cars": cars,
        "humans": humans,
        "vegetation": vegetation,
    }


def normalize_street_data(cfg: dict) -> dict:
    cfg = deepcopy(cfg)

    for side in ("left", "right"):
        side_cfg = cfg["sides"][side]
        for key in ("driveway", "bikepath", "footpath"):
            node = side_cfg[key]
            node["width"] = max(0.0, float(node.get("width", 0.0) or 0.0))
            node["present"] = bool(node.get("present", False)) and node["width"] > 0
            if not node["present"]:
                node["width"] = 0.0

        for key in ("buildings", "trees"):
            node = side_cfg[key]
            node["count"] = max(0, int(node.get("count", 0) or 0))
            node["present"] = bool(node.get("present", False)) and node["count"] > 0
            if not node["present"]:
                node["count"] = 0

    median = cfg["median"]
    median["width"] = max(0.0, float(median.get("width", 0.0) or 0.0))
    median["present"] = bool(median.get("present", False)) and median["width"] > 0
    if not median["present"]:
        median["width"] = 0.0
        median["trees"] = {"present": False, "count": 0}
    else:
        median["trees"]["count"] = max(0, int(median["trees"].get("count", 0) or 0))
        median["trees"]["present"] = bool(median["trees"].get("present", False)) and median["trees"]["count"] > 0
        if not median["trees"]["present"]:
            median["trees"]["count"] = 0

    sp = cfg["street_parking"]
    if not sp.get("present", False):
        cfg["street_parking"] = {"present": False}
    else:
        w = sp.get("width_m_per_side", {"left": 0.0, "right": 0.0})
        w["left"] = max(0.0, float(w.get("left", 0.0) or 0.0))
        w["right"] = max(0.0, float(w.get("right", 0.0) or 0.0))
        sp["width_m_per_side"] = w
        if w["left"] > 0 and w["right"] > 0:
            sp["sides"] = "both"
        elif w["left"] > 0:
            sp["sides"] = "left"
        elif w["right"] > 0:
            sp["sides"] = "right"
        else:
            cfg["street_parking"] = {"present": False}

    for key in ("parked_cars", "cars", "humans"):
        cfg[key]["count"] = max(0, int(cfg[key].get("count", 0) or 0))
        cfg[key]["present"] = bool(cfg[key].get("present", False)) and cfg[key]["count"] > 0
        if not cfg[key]["present"]:
            cfg[key]["count"] = 0

    cfg["vegetation"]["density"] = max(0.0, float(cfg["vegetation"].get("density", 0.0) or 0.0))
    cfg["vegetation"]["present"] = bool(cfg["vegetation"].get("present", False)) and cfg["vegetation"]["density"] > 0
    if not cfg["vegetation"]["present"]:
        cfg["vegetation"]["density"] = 0.0

    cfg["length"] = max(1.0, float(cfg.get("length", 50.0) or 50.0))
    cfg["schemaVersion"] = 3
    return cfg
