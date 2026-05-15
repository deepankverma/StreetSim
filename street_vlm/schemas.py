"""Pydantic schemas for vision output and final street_data.json."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Confidence = float


class WidthEstimate(BaseModel):
    present: bool
    width_m: float = Field(ge=0)
    confidence: Confidence = Field(ge=0, le=1)
    evidence: Optional[str] = None


class CountEstimate(BaseModel):
    present: bool
    count_visible: int = Field(ge=0)
    confidence: Confidence = Field(ge=0, le=1)
    evidence: Optional[str] = None


class ParkingEstimate(BaseModel):
    present: bool
    bay_width_m: float = Field(ge=0)
    count_visible: int = Field(ge=0)
    confidence: Confidence = Field(ge=0, le=1)
    evidence: Optional[str] = None


class SideVision(BaseModel):
    driveway: WidthEstimate
    bikepath: WidthEstimate
    footpath: WidthEstimate
    buildings: CountEstimate
    trees: CountEstimate
    parking: ParkingEstimate


class MedianVision(BaseModel):
    present: bool
    width_m: float = Field(ge=0)
    trees_present: bool
    tree_count_visible: int = Field(ge=0)
    confidence: Confidence = Field(ge=0, le=1)
    evidence: Optional[str] = None


class DynamicVision(BaseModel):
    parked_cars_visible: int = Field(ge=0)
    moving_cars_visible: int = Field(ge=0)
    pedestrians_visible: int = Field(ge=0)


class VegetationVision(BaseModel):
    present: bool
    density_score_0_100: int = Field(ge=0, le=100)
    confidence: Confidence = Field(ge=0, le=1)
    evidence: Optional[str] = None


class VisionStreetSchema(BaseModel):
    scene_type: Literal["urban_street", "suburban_street", "residential_street", "unknown"] = "urban_street"
    camera_view: Literal["street_level", "elevated", "unknown"] = "street_level"
    left_side: SideVision
    right_side: SideVision
    median: MedianVision
    dynamic: DynamicVision
    vegetation: VegetationVision
    street_length_m_estimate: Optional[float] = Field(default=None, ge=0)
    global_notes: List[str] = Field(default_factory=list)
    unresolvable: List[str] = Field(default_factory=list)


class PresenceWidth(BaseModel):
    present: bool
    width: float = Field(ge=0)


class PresenceCount(BaseModel):
    present: bool
    count: int = Field(ge=0)


class SideConfig(BaseModel):
    driveway: PresenceWidth
    bikepath: PresenceWidth
    footpath: PresenceWidth
    buildings: PresenceCount
    trees: PresenceCount


class MedianConfig(BaseModel):
    present: bool
    width: float = Field(ge=0)
    trees: PresenceCount


class StreetParkingConfig(BaseModel):
    present: bool
    sides: Literal["none", "left", "right", "both"] = "none"
    width_m_per_side: Optional[Dict[str, float]] = None
    min_driveway_width: Optional[float] = 3.0


class ParkedCarsConfig(BaseModel):
    present: bool
    count: int = Field(ge=0)
    seed: int = 0


class CarsConfig(BaseModel):
    present: bool
    count: int = Field(ge=0)
    scale: float = 10.0


class HumansConfig(BaseModel):
    present: bool
    count: int = Field(ge=0)
    scale: float = 5.0
    z_offset: float = 1.2


class VegetationConfig(BaseModel):
    present: bool
    density: float = Field(ge=0)


class StreetDataConfig(BaseModel):
    schemaVersion: int = 3
    length: float = Field(ge=1)
    sides: Dict[str, SideConfig]
    median: MedianConfig
    street_parking: dict
    parked_cars: ParkedCarsConfig
    cars: CarsConfig
    humans: HumansConfig
    vegetation: VegetationConfig
