"""Unit tests for the LoD1 meshing utilities."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pyvista as pv
from shapely.geometry import MultiPolygon, Point, Polygon

from lod1 import (
    estimate_building_heights,
    extrude_polygon_to_shell,
    lod1_buildings_to_meshes,
    polygon_with_holes_to_surface,
)


def test_extrude_polygon_to_shell_is_watertight() -> None:
    square = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    mesh = extrude_polygon_to_shell(square, height=8.0, z0=1.5)

    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_cells > 0
    assert mesh.n_open_edges == 0
    assert np.isclose(mesh.bounds[4], 1.5)
    assert np.isclose(mesh.bounds[5], 9.5)


def test_polygon_with_hole_has_no_triangles_in_courtyard() -> None:
    outer = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)]
    hole = [(2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)]
    polygon = Polygon(outer, holes=[hole])

    surface = polygon_with_holes_to_surface(polygon, z0=0.0)
    faces = surface.faces.reshape(-1, 4)
    hole_poly = Polygon(hole)

    for _, i0, i1, i2 in faces:
        tri_pts = surface.points[[i0, i1, i2], :2]
        centroid = np.mean(tri_pts, axis=0)
        assert not hole_poly.contains(Point(centroid))


def test_estimate_building_heights_uses_levels_and_fallback() -> None:
    geoms = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
    ]
    gdf = gpd.GeoDataFrame(
        {
            "geometry": geoms,
            "height_m": [20.0, np.nan, np.nan],
            "building:levels": [np.nan, 3.0, np.nan],
        },
        crs="EPSG:3857",
    )

    estimated = estimate_building_heights(gdf, meters_per_level=4.0)
    heights = estimated["height_final_m"].tolist()
    sources = estimated["height_final_source"].tolist()

    assert np.isclose(heights[0], 20.0)
    assert sources[0] == "height_column"
    assert np.isclose(heights[1], 12.0)
    assert sources[1] == "levels_estimate"
    assert np.isclose(heights[2], 16.0)
    assert sources[2] == "global_average"


def test_lod1_buildings_to_meshes_handles_multipolygons_and_height_policy() -> None:
    def _square(lon: float, lat: float, size_deg: float) -> Polygon:
        return Polygon(
            [
                (lon, lat),
                (lon + size_deg, lat),
                (lon + size_deg, lat + size_deg),
                (lon, lat + size_deg),
            ]
        )

    multi = MultiPolygon([_square(0.0, 0.0, 0.001), _square(0.002, 0.0, 0.001)])
    with_levels = _square(0.01, 0.0, 0.0015)
    fallback = _square(0.02, 0.0, 0.001)

    gdf = gpd.GeoDataFrame(
        {
            "geometry": [multi, with_levels, fallback],
            "height": [12.0, np.nan, np.nan],
            "levels": [np.nan, 2.0, np.nan],
            "building_id": ["multi", "levels", "fallback"],
        },
        crs="EPSG:4326",
    )

    meshes = lod1_buildings_to_meshes(gdf, z0=2.0, combine=False)
    assert len(meshes) == 4  # two parts from the MultiPolygon.

    labels = [label for label, _ in meshes]
    assert labels[:2] == ["multi_part_1", "multi_part_2"]
    assert set(labels) == {"multi_part_1", "multi_part_2", "levels", "fallback"}

    # Validate heights from policy: explicit height (12), derived from levels (2*3), and global average fallback (9).
    heights = {}
    for label, mesh in meshes:
        heights[label] = float(mesh.cell_data["height"][0])
        assert mesh.n_open_edges == 0

    assert np.isclose(heights["multi_part_1"], 12.0)
    assert np.isclose(heights["multi_part_2"], 12.0)
    assert np.isclose(heights["levels"], 6.0)
    assert np.isclose(heights["fallback"], 9.0)

    # Combined scene exposes building_id scalars and remains watertight.
    scene = lod1_buildings_to_meshes(gdf, z0=2.0, combine=True)
    assert isinstance(scene, pv.PolyData)
    assert scene.n_cells > 0
    assert scene.n_open_edges == 0
    assert "building_id" in scene.cell_data
    assert np.unique(scene.cell_data["building_id"]).size == 4
