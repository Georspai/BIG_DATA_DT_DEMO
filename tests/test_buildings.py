"""Quick test script that fetches buildings around Thessaloniki (100 m radius)."""

from __future__ import annotations

import geopandas as gpd
import osmnx as ox
from shapely.geometry.base import BaseGeometry

from osmnx_ingest import IngestConfig, ingest_osm


def _circle_from_place(place: str, radius_m: float) -> BaseGeometry:
    """Create a circular polygon in WGS84 with the given radius (in meters)."""
    area_gdf = ox.geocode_to_gdf(place)
    if area_gdf.empty:
        raise ValueError(f"Could not resolve '{place}' to a geometry.")

    centroid = area_gdf.geometry.iloc[0].centroid
    centroid_gdf = gpd.GeoSeries([centroid], crs="EPSG:4326")
    utm_crs = centroid_gdf.estimate_utm_crs()
    centroid_utm = centroid_gdf.to_crs(utm_crs)
    circle = centroid_utm.buffer(radius_m)
    return circle.to_crs("EPSG:4326").iloc[0]


def main() -> None:
    place = "Thessaloniki, Greece"
    radius_m = 100.0
    polygon = _circle_from_place(place, radius_m)

    cfg = IngestConfig(height_policy="levels_x_3m", crs_out="utm")
    result = ingest_osm(polygon=polygon, cfg=cfg)

    print(f"Fetched {len(result.buildings)} buildings within {radius_m} m of {place}.")
    print(result.buildings.head())


if __name__ == "__main__":
    main()
