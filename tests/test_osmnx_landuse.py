"""Tests for the landuse ingestion helper."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from osmnx_ingest import IngestConfig
from osmnx_ingest.workflow import get_landuse


def test_get_landuse_filters_to_bbox_and_sets_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure only polygons fully inside the area bbox are kept, with osm ids assigned."""

    area = box(0.0, 0.0, 1.0, 1.0)
    polygon_inside = Polygon([(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)])
    polygon_outside = Polygon([(1.2, 1.2), (1.4, 1.2), (1.4, 1.4), (1.2, 1.4)])

    gdf = gpd.GeoDataFrame(
        {
            "geometry": [polygon_inside, polygon_outside],
            "name": ["Park In", "Park Out"],
            "landuse": ["residential", "residential"],
        },
        crs="EPSG:4326",
    )
    gdf.index = pd.MultiIndex.from_tuples([("way", 1), ("way", 2)])
    expected_crs = gdf.estimate_utm_crs()

    def _features_from_polygon(*args, **kwargs):  # type: ignore[no-untyped-def]
        return gdf

    monkeypatch.setattr(
        "osmnx_ingest.workflow.ox.features_from_polygon", _features_from_polygon
    )

    cfg = IngestConfig(crs_out="utm")
    landuse = get_landuse(area, cfg)

    assert len(landuse) == 1
    assert landuse.iloc[0]["name"] == "Park In"
    assert landuse.iloc[0]["osm_id"] == "way/1"
    assert list(landuse.columns) == [
        "osm_id",
        "geometry",
        "name",
        "landuse",
        "natural",
        "leisure",
        "source",
    ]
    assert landuse.crs == expected_crs


def test_get_landuse_returns_empty_layer_when_no_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If no polygon features are returned or all filtered out, we should get an empty layer."""

    polygon_outside = Polygon([(2.0, 2.0), (2.5, 2.0), (2.5, 2.5), (2.0, 2.5)])
    outside_gdf = gpd.GeoDataFrame(
        {
            "geometry": [polygon_outside],
            "landuse": ["forest"],
        },
        crs="EPSG:4326",
    )

    def _features_from_polygon(*args, **kwargs):  # type: ignore[no-untyped-def]
        return outside_gdf

    monkeypatch.setattr(
        "osmnx_ingest.workflow.ox.features_from_polygon", _features_from_polygon
    )

    landuse = get_landuse(box(0, 0, 1, 1), IngestConfig())

    assert landuse.empty
    assert landuse.crs and landuse.crs.to_string() == "EPSG:4326"

    empty_gdf = gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries([], crs="EPSG:4326")},
        crs="EPSG:4326",
    )

    def _features_empty(*args, **kwargs):  # type: ignore[no-untyped-def]
        return empty_gdf

    monkeypatch.setattr(
        "osmnx_ingest.workflow.ox.features_from_polygon", _features_empty
    )

    landuse = get_landuse(box(0, 0, 1, 1), IngestConfig())

    assert landuse.empty
    assert landuse.crs and landuse.crs.to_string() == "EPSG:4326"

