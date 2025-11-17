"""Workflow orchestration for the osmnx ingest module."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from dtcc import Bounds

from .config import IngestConfig

from .utils import (
    apply_osmnx_settings,
    empty_generic_layer,
    _empty_landuse_layer,
    _clean_polygonal_layer,
    reproject_if_requested,
    format_osm_index
)

def get_landuse(bounding_box:  Bounds | Polygon, config: IngestConfig) -> gpd.GeoDataFrame:
    """Fetch polygonal landuse features within the requested bounding box."""

    apply_osmnx_settings(config)
    bbox_poly = box(*bounding_box.bounds)
    tags = {
        "landuse": True,
        "natural": [
            "wood",
            "scrub",
            "grassland",
            "heath",
            "fell",
            "bare_rock",
            "sand",
        ],
        "leisure": [
            "park",
            "pitch",
            "garden",
            "golf_course",
            "recreation_ground",
            "playground",
            "sports_centre",
            "track",
        ],
    }

    gdf = ox.features_from_polygon(bounding_box, tags=tags)
    if gdf.empty:
        return _empty_landuse_layer("EPSG:4326")

    gdf = gdf.copy()
    gdf["osm_id"] = gdf.index.map(format_osm_index)
    gdf = _clean_polygonal_layer(gdf)
    if gdf.empty:
        return _empty_landuse_layer("EPSG:4326")

    gdf = gdf[gdf.geometry.intersects(bbox_poly)].copy()

    gdf["geometry"] = gdf.geometry.intersection(bbox_poly)
    gdf = gdf[~gdf.is_empty]
    
    if gdf.empty:
        return _empty_landuse_layer(gdf.crs)

    required_defaults = {
        "name": pd.NA,
        "landuse": pd.NA,
        "natural": pd.NA,
        "leisure": pd.NA,
        "source": pd.NA,
    }
    for column, default in required_defaults.items():
        if column not in gdf.columns:
            gdf[column] = default

    ordered_cols = [
        "osm_id",
        "geometry",
        "name",
        "landuse",
        "natural",
        "leisure",
        "source",
    ]

    gdf = gdf.reset_index(drop=True)
    gdf = reproject_if_requested(gdf, config)

    if config.crs_out is None:
        config.crs_out = "EPSG:4326"
    return gdf[ordered_cols].to_crs( config.crs_out)