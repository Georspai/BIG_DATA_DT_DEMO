"""Configuration objects and type aliases for the OSMnx ingest pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Union


RoadNetworkTypeLiteral = Literal["drive", "walk", "bike", "all", "all_private"]



@dataclass
class IngestConfig:
    """User-facing configuration for the ingest workflow."""

    roadnetwork_type: RoadNetworkTypeLiteral = "drive"
    use_cache: bool = True
    timeout: int = 180
    crs_out: Optional[str] = None  # None -> EPSG:4326, "utm" -> auto per layer
    random_seed: Optional[int] = 42
    simplify_roads: bool = True
    add_speeds_times: bool = True
    polygonize_water: bool = True