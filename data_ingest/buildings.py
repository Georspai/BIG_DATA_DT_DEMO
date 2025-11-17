"""Building and building part ingestion helpers."""

from __future__ import annotations

import math
import re
from typing import Dict, Tuple, Optional

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from shapely.geometry import Polygon, box

from dtcc import Bounds

from .config import IngestConfig
from .utils import apply_osmnx_settings


def get_buildings(
    bounding_box: Bounds | Polygon,
    config: IngestConfig,
    tags: Optional[dict[str, bool | str | list[str]]] = None,
    assign_uid: bool = True
):
    apply_osmnx_settings(config)

    # Handle Bounds → Polygon
    if isinstance(bounding_box, Bounds):
        bounding_box = box(
            minx=bounding_box.xmin,
            miny=bounding_box.ymin,
            maxx=bounding_box.xmax,
            maxy=bounding_box.ymax,
        )

    if tags is None:
        tags = {"building": True}

    # Query OSMnx
    gdf = ox.features_from_polygon(bounding_box, tags=tags)

    # Ensure CRS
    if config.crs_out is None:
        config.crs_out = "EPSG:4326"

    gdf = gdf.to_crs(config.crs_out)

    # Replace tuple index with unique ID
    if assign_uid:
        gdf = _assign_osm_uid(gdf)

    return gdf


def _format_osm_uid(osm_index: tuple[str, int]) -> str:
    """Convert OSMnx index tuples into a unique, stable string ID.

    Example:
        ('relation', 13080752) → 'relation_13080752'
    """
    if not isinstance(osm_index, tuple) or len(osm_index) != 2:
        return str(osm_index)  # fallback

    element_type, osm_id = osm_index
    return f"{element_type}_{int(osm_id)}"


def _assign_osm_uid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Takes a GeoDataFrame returned by OSMnx and replaces its index 
    (e.g., ('relation', 13080752)) with a new column `uid` that stores
    a unique string identifier.
    """
    # Make a copy to avoid modifying original
    gdf = gdf.copy()

    # Create new UID column
    gdf["uid"] = [_format_osm_uid(idx) for idx in gdf.index]

    # Reset index to a simple RangeIndex
    gdf = gdf.reset_index(drop=True)

    return gdf
