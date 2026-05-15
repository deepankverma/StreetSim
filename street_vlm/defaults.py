"""Default policy values for mapping vision output into street_data.json."""

from __future__ import annotations

DEFAULT_POLICY = {
    "default_length_m": 50.0,
    "min_confidence_keep": 0.45,
    "round_width_to": 0.5,
    "max_buildings_per_side": 12,
    "max_trees_per_side": 20,
    "minimum_humans_total": 1, ### configure later
    "visible_to_total_mode": "visible_only",  # visible_only | multiplier | clamp_minimum
    "count_multiplier": {
        "buildings": 1.0,
        "trees": 1.0,
        "parked_cars": 1.0,
        "moving_cars": 1.0,
        "pedestrians": 1.0,
        "median_trees": 1.0,
    },
    "minimum_counts_when_present": {
        "buildings": 1,
        "trees": 1,
        "parked_cars": 1,
        "moving_cars": 1,
        "pedestrians": 1,
        "median_trees": 1,
    },
    "parking_min_driveway_width": 3.0,
    "cars_scale": 10.0,
    "humans_scale": 5.0,
    "humans_z_offset": 1.2,
    "vegetation_density_divisor": 100.0,
}
