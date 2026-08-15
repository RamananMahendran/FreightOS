from pydantic import BaseModel, Field
from typing import Literal

class NodeSchema(BaseModel):
    id: str = Field(..., example="STN_CHENNAI_CENTRAL")
    name: str = Field(..., example="Chennai Central Yard")
    node_type: Literal["Station", "Siding", "Terminal", "Port", "Industry"]
    capacity: int = Field(default=50, description="Max train holding capacity")

class TrackSegmentSchema(BaseModel):
    source_id: str = Field(..., example="STN_CHENNAI_CENTRAL")
    target_id: str = Field(..., example="STN_ARAKKONAM_JCT")
    segment_id: str = Field(..., example="SEG_CHN_AJJ_01")
    length_km: float = Field(..., gt=0, example=68.5)
    speed_limit_kmh: float = Field(..., gt=0, example=110.0)
    max_axle_load_tons: float = Field(..., gt=0, example=22.5)
    traversing_capacity: int = Field(..., gt=0, example=15, description="Max concurrent trains allowed")