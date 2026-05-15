"""Prompt templates for multi-pass street extraction."""

from __future__ import annotations

COARSE_PROMPT = """You are extracting a structured street-layout description from one street-view image.

Return JSON only. Do not add prose.

Task:
1. Analyze the left side, right side, and center median separately.
2. Decide whether the following are visibly present on each side:
   - driveway / carriageway [street where vehicles move] adjacent to the center
   - bike lane
   - footpath / sidewalk
   - buildings
   - trees
   - on-street parking
3. For the whole frame, count visible:
   - parked cars
   - moving cars
   - pedestrians
4. Decide whether a median exists.
5. Decide whether vegetation is generally present.

Rules:
- Use only what is visually supported by the image.
- If uncertain, still make a best guess and attach a confidence score from 0 to 1.
- Do not infer file paths, render settings, pipeline stages, Blender settings, or anything not visible.
- Do not invent hidden geometry outside the frame.
- Width fields must be numeric, even if they remain 0 at this stage.

Return exactly this JSON shape:
{
  "scene_type": "urban_street|suburban_street|residential_street|unknown",
  "camera_view": "street_level|elevated|unknown",
  "left_side": {
    "driveway": {"present": true, "width_m": 0, "confidence": 0, "evidence": ""},
    "bikepath": {"present": true, "width_m": 0, "confidence": 0, "evidence": ""},
    "footpath": {"present": true, "width_m": 0, "confidence": 0, "evidence": ""},
    "buildings": {"present": true, "count_visible": 0, "confidence": 0, "evidence": ""},
    "trees": {"present": true, "count_visible": 0, "confidence": 0, "evidence": ""},
    "parking": {"present": true, "bay_width_m": 0, "count_visible": 0, "confidence": 0, "evidence": ""}
  },
  "right_side": {
    "driveway": {"present": true, "width_m": 0, "confidence": 0, "evidence": ""},
    "bikepath": {"present": true, "width_m": 0, "confidence": 0, "evidence": ""},
    "footpath": {"present": true, "width_m": 0, "confidence": 0, "evidence": ""},
    "buildings": {"present": true, "count_visible": 0, "confidence": 0, "evidence": ""},
    "trees": {"present": true, "count_visible": 0, "confidence": 0, "evidence": ""},
    "parking": {"present": true, "bay_width_m": 0, "count_visible": 0, "confidence": 0, "evidence": ""}
  },
  "median": {
    "present": true,
    "width_m": 0,
    "trees_present": true,
    "tree_count_visible": 0,
    "confidence": 0,
    "evidence": ""
  },
  "dynamic": {
    "parked_cars_visible": 0,
    "moving_cars_visible": 0,
    "pedestrians_visible": 0
  },
  "vegetation": {
    "present": true,
    "density_score_0_100": 0,
    "confidence": 0,
    "evidence": ""
  },
  "street_length_m_estimate": null,
  "global_notes": [],
  "unresolvable": []
}"""

REFINE_PROMPT = """You are refining a previously extracted street schema from one street-view image.

Return JSON only.

You will receive:
1. an image
2. a draft JSON extraction

Your job:
- refine approximate metric widths in meters for features that are present
- correct visible counts if the draft looks inconsistent
- keep absent features at width 0
- improve confidence and evidence fields

Guidelines for metric estimation:
- Use common urban design priors when exact calibration is impossible.
- Typical footpaths are often around 1.5 to 5 m.
- Typical bike lanes are often around 1.2 to 2.5 m.
- Typical parking bays are often around 2.2 to 3.0 m.
- Typical single carriageway side widths vary widely; estimate conservatively.
- Median widths are often narrow unless clearly landscaped.
- If a feature is not visible, do not fabricate it.

Important:
- Preserve the same JSON structure.
- Do not add any new top-level keys.
- If you are unsure, lower confidence instead of hallucinating precision."""

CHECK_PROMPT = """You are validating a street-extraction JSON object.

Return JSON only.

Enforce these rules:
- if a feature is absent, its width must be 0
- if parking is absent, bay_width_m must be 0 and count_visible must be 0
- if buildings are absent, count_visible must be 0
- if trees are absent, count_visible must be 0
- if median is absent, width_m must be 0 and tree_count_visible must be 0 and trees_present must be false
- all counts must be integers >= 0
- all widths must be >= 0
- all confidence values must be between 0 and 1
- keep only the allowed keys from the original schema

Do not add commentary. Return the corrected JSON only."""
