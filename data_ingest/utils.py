"""Utility helpers shared across ingest modules."""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd
import osmnx as ox

from shapely import box,Point,Polygon
from typing import Any
from .config import IngestConfig

def to_wgs84(poly: Polygon, crs: CRSType) -> Polygon:
    """Reproject a polygon from given CRS to EPSG:4326."""
    gs = gpd.GeoSeries([poly], crs=crs)
    return gs.to_crs("EPSG:4326").iloc[0]

def from_wgs84(poly_wgs84: Polygon, target_crs: CRSType) -> Polygon:
    """Reproject a polygon from EPSG:4326 to target CRS."""
    gs = gpd.GeoSeries([poly_wgs84], crs="EPSG:4326")
    return gs.to_crs(target_crs).iloc[0]

def apply_osmnx_settings(cfg: IngestConfig) -> None:
    """Synchronize osmnx global settings with the current ingest config."""

    ox.settings.use_cache = cfg.use_cache
    ox.settings.requests_timeout = cfg.timeout



def square_from_place(place: str, side_m: float) -> Any:
    """Create a square polygon in WGS84 centered around the place centroid."""

    area_gdf = ox.geocode_to_gdf(place)
    if area_gdf.empty:
        raise ValueError(f"Could not resolve '{place}' to a geometry.")

    centroid = area_gdf.geometry.iloc[0].centroid
    centroid_gdf = gpd.GeoSeries([centroid], crs="EPSG:4326")
    utm_crs = centroid_gdf.estimate_utm_crs()

    print("Estimated CRS for the provided area: ",utm_crs )
    centroid_utm = centroid_gdf.to_crs(utm_crs)
    half_side = side_m / 2.0
    centroid_pt = centroid_utm.iloc[0]
    rect = box(
        centroid_pt.x - half_side,
        centroid_pt.y - half_side,
        centroid_pt.x + half_side,
        centroid_pt.y + half_side,
    )
    return gpd.GeoSeries([rect], crs=utm_crs).to_crs("EPSG:4326").iloc[0]

from shapely.geometry import Point, box
from pyproj import CRS
from typing import Union, Any
import geopandas as gpd

CRSType = Union[str, CRS]

def square_from_point(
    x: float,
    y: float,
    input_crs: CRSType,
    side_m: float,
    output_crs: str = "EPSG:4326",
) -> Any:
    """
    Create a square polygon centered on the point with given side length in meters.
    The result is returned in `output_crs` (default: EPSG:4326 for Overpass/OSMnx).
    """

    if side_m <= 0:
        raise ValueError("side_m must be positive to form a bounding box.")

    point_series = gpd.GeoSeries([Point(x, y)], crs=input_crs)
    if point_series.crs is None:
        raise ValueError("input_crs must be provided to construct the bounding box.")

    crs_obj = point_series.crs
    axis_units = {axis.unit_name.lower() for axis in crs_obj.axis_info if axis.unit_name}

    # Work in a metric CRS
    if crs_obj.is_projected and any(
        unit.startswith("metre") or unit.startswith("meter") for unit in axis_units
    ):
        metric_series = point_series
        metric_crs = crs_obj
    else:
        # Go to WGS84, then pick a local UTM zone
        wgs_series = point_series.to_crs("EPSG:4326")
        utm_crs = wgs_series.estimate_utm_crs()
        metric_series = wgs_series.to_crs(utm_crs)
        metric_crs = utm_crs

    half_side = side_m / 2.0
    metric_point = metric_series.iloc[0]
    rect = box(
        metric_point.x - half_side,
        metric_point.y - half_side,
        metric_point.x + half_side,
        metric_point.y + half_side,
    )

    # Return in the requested output CRS (EPSG:4326 for Overpass or anything else)
    return gpd.GeoSeries([rect], crs=metric_crs).to_crs(output_crs).iloc[0]

def empty_generic_layer(crs: Optional[str]) -> gpd.GeoDataFrame:
    """Create an empty GeoDataFrame with only a geometry column."""

    geometry = gpd.GeoSeries([], crs=crs or "EPSG:4326")
    return gpd.GeoDataFrame({"geometry": geometry}, geometry=geometry, crs=geometry.crs)

def _empty_landuse_layer(crs: Optional[str]) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame matching the landuse schema."""

    data = {
        "osm_id": pd.Series(dtype="object"),
        "name": pd.Series(dtype="object"),
        "landuse": pd.Series(dtype="object"),
        "natural": pd.Series(dtype="object"),
        "leisure": pd.Series(dtype="object"),
        "source": pd.Series(dtype="object"),
    }
    geometry = gpd.GeoSeries([], crs=crs or "EPSG:4326")
    return gpd.GeoDataFrame(data, geometry=geometry, crs=geometry.crs)


def _clean_polygonal_layer(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep polygonal features, explode multipart geometries, and fix invalid ones."""

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        return gdf

    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[~gdf.geometry.is_empty]
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.is_valid]
    gdf = gdf.reset_index(drop=True)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def _empty_water_layer(crs: Optional[str]) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame matching the water schema."""

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


def reproject_if_requested(
    gdf: gpd.GeoDataFrame, cfg: IngestConfig
) -> gpd.GeoDataFrame:
    """Reproject to an on-the-fly UTM CRS if requested."""

    if cfg.crs_out != "utm" or gdf.empty:
        return gdf
    try:
        target_crs = gdf.estimate_utm_crs()
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("Unable to estimate UTM CRS for projection.") from exc
    if target_crs is None:
        raise RuntimeError("estimate_utm_crs returned None for the layer.")
    return gdf.to_crs(target_crs)


def format_osm_index(osm_index: object) -> str:
    """Convert OSMnx multi-index values to canonical osm_id strings."""

    if isinstance(osm_index, tuple) and len(osm_index) >= 2:
        prefix, osm_id = osm_index[0], osm_index[1]
        return f"{prefix}/{osm_id}"
    return str(osm_index)

