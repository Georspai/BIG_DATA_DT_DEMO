from __future__ import annotations

from typing import Optional

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from dtcc import Bounds

from .config import IngestConfig

from .utils import (
    apply_osmnx_settings,
    _clean_polygonal_layer,
    format_osm_index,
)


def get_water(
    bounding_box: Bounds | BaseGeometry,
    config: IngestConfig,
    include_seas: bool = False,
) -> gpd.GeoDataFrame:
    """Fetch polygonal water bodies within the requested bounding box."""

    apply_osmnx_settings(config)
    bbox_geom = _normalize_bounding_geometry(bounding_box)
    bbox_poly = box(*bbox_geom.bounds)

    required_defaults = {
        "name": pd.NA,
        "natural": pd.NA,
        "water": pd.NA,
        "waterway": pd.NA,
        "landuse": pd.NA,
        "source": pd.NA,
    }
    ordered_cols = [
        "osm_id",
        "geometry",
        "name",
        "natural",
        "water",
        "waterway",
        "landuse",
        "source",
    ]

    tags = {
        "natural": ["water", "wetland", "bay", "strait", "shoal", "sea"],
        "water": True,
        "waterway": ["riverbank", "dock"],
        "landuse": ["basin", "reservoir", "salt_pond"],
        "place": "sea",
    }

    layers: list[gpd.GeoDataFrame] = []

    base_gdf = ox.features_from_polygon(bbox_geom, tags=tags)
    base_layer = _prepare_water_layer(base_gdf, bbox_poly, required_defaults)
    if not base_layer.empty:
        layers.append(base_layer)

    if include_seas:
        sea_raw = _get_sea_polygons(bbox_geom)
        sea_layer = _prepare_water_layer(sea_raw, bbox_poly, required_defaults)
        if not sea_layer.empty:
            layers.append(sea_layer)

    if not layers:
        return _empty_water_layer("EPSG:4326")

    combined = gpd.GeoDataFrame(
        pd.concat(layers, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )
    combined = combined.loc[:, ordered_cols]
    combined = combined.reset_index(drop=True)
    
    if config.crs_out is None:
        config.crs_out = "EPSG:4326"

    return combined.to_crs(config.crs_out)


def _prepare_water_layer(
    raw_gdf: Optional[gpd.GeoDataFrame],
    bbox_poly: BaseGeometry,
    required_defaults: dict[str, object],
) -> gpd.GeoDataFrame:
    if raw_gdf is None or raw_gdf.empty:
        return _empty_water_layer("EPSG:4326")

    gdf = raw_gdf.copy()
    gdf["osm_id"] = gdf.index.map(format_osm_index)
    gdf = _clean_polygonal_layer(gdf)
    if gdf.empty:
        return gdf

    gdf = gdf[gdf.geometry.intersects(bbox_poly)].copy()
    if gdf.empty:
        return gdf

    gdf["geometry"] = gdf.geometry.intersection(bbox_poly)
    gdf = gdf[~gdf.is_empty]
    if gdf.empty:
        return gdf

    for column, default in required_defaults.items():
        if column not in gdf.columns:
            gdf[column] = default

    return gdf.reset_index(drop=True)


def _get_sea_polygons(bbox_geom: BaseGeometry) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox_geom.bounds
    sea_tags = {
        "natural": ["sea", "water", "bay"],
        "place": "sea",
    }
    sea_gdf = ox.features_from_bbox(maxy, miny, maxx, minx, tags=sea_tags)
    if sea_gdf.empty:
        return sea_gdf

    poly_mask = sea_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    sea_mask = (
        (sea_gdf.get("natural") == "sea")
        | ((sea_gdf.get("natural") == "water") & (sea_gdf.get("water") == "sea"))
        | (sea_gdf.get("place") == "sea")
        | (sea_gdf.get("natural") == "bay")
    )
    return sea_gdf[poly_mask & sea_mask].copy()


def _normalize_bounding_geometry(
    bounding_box: Bounds | BaseGeometry,
) -> BaseGeometry:
    if isinstance(bounding_box, Bounds):
        xmin = min(bounding_box.xmin, bounding_box.xmax)
        xmax = max(bounding_box.xmin, bounding_box.xmax)
        ymin = min(bounding_box.ymin, bounding_box.ymax)
        ymax = max(bounding_box.ymin, bounding_box.ymax)
        return box(xmin, ymin, xmax, ymax)
    if isinstance(bounding_box, BaseGeometry):
        return bounding_box
    raise TypeError("bounding_box must be a dtcc.Bounds or shapely geometry")


def _empty_water_layer(crs: Optional[str]) -> gpd.GeoDataFrame:
    data = {
        "osm_id": pd.Series(dtype="object"),
        "name": pd.Series(dtype="object"),
        "natural": pd.Series(dtype="object"),
        "water": pd.Series(dtype="object"),
        "waterway": pd.Series(dtype="object"),
        "landuse": pd.Series(dtype="object"),
        "source": pd.Series(dtype="object"),
    }
    geometry = gpd.GeoSeries([], crs=crs or "EPSG:4326")
    return gpd.GeoDataFrame(data, geometry=geometry, crs=geometry.crs)
