"""Utility functions for converting planar building footprints to LoD1 shells."""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Iterator, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyvista as pv
from pyproj import CRS
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import triangulate

logger = logging.getLogger(__name__)

DEFAULT_METRIC_CRS = "EPSG:3857"
DEFAULT_HEIGHT_METERS = 10.0
DEFAULT_METERS_PER_LEVEL = 3.0


def estimate_building_heights(
    buildings: gpd.GeoDataFrame,
    meters_per_level: float = DEFAULT_METERS_PER_LEVEL,
    height_columns: tuple[str, ...] = ("height_m", "height"),
    level_columns: tuple[str, ...] = ("building:levels", "levels"),
    roof_height_columns: tuple[str, ...] = ("roof:height",),
    roof_level_columns: tuple[str, ...] = ("roof:levels",),
) -> gpd.GeoDataFrame:
    """
    Estimate an LoD1-ready height for each building footprint.

    Heights are determined by examining the supplied columns in priority order:

    1. Use `height_columns` (e.g., `height_m`, `height`) when present.
    2. Otherwise, convert `level_columns` to meters via `meters_per_level`.
    3. If still missing but a roof height/level exists, add that roof amount to the dataset
       average height.
    4. Finally, fall back to the dataset average (or `DEFAULT_HEIGHT_METERS` if nothing exists).

    Parameters
    ----------
    buildings:
        GeoDataFrame of building footprints and their attributes.
    meters_per_level:
        Conversion factor from floor counts to meters.
    height_columns, level_columns, roof_height_columns, roof_level_columns:
        Ordered tuples listing preferred columns for each measurement type. The first
        non-null numeric value found across the tuple is used.

    Returns
    -------
    GeoDataFrame
        Copy of the input with `height_final_m` and `height_final_source` columns describing
        the resolved heights and the rule that produced them.
    """

    if buildings is None:
        raise ValueError("buildings GeoDataFrame cannot be None.")

    gdf = buildings.copy()
    if gdf.empty:
        gdf["height_final_m"] = pd.Series(dtype="float64")
        gdf["height_final_source"] = pd.Series(dtype="object")
        return gdf

    def _first_numeric_columns(columns: tuple[str, ...]) -> pd.Series:
        values = pd.Series(np.nan, index=gdf.index, dtype="float64")
        for col in columns:
            if col not in gdf.columns:
                continue
            numeric = pd.to_numeric(gdf[col], errors="coerce")
            values = values.where(values.notna(), numeric)
        return values.astype("float64")

    heights = _first_numeric_columns(height_columns)
    levels = _first_numeric_columns(level_columns)
    roof_heights = _first_numeric_columns(roof_height_columns)
    roof_levels = _first_numeric_columns(roof_level_columns)

    final_height = heights.copy()
    final_source = pd.Series(pd.NA, index=gdf.index, dtype="object")
    height_mask = final_height.notna()
    final_source.loc[height_mask] = "height_column"

    level_estimate = levels * float(meters_per_level)
    use_levels = final_height.isna() & level_estimate.notna()
    final_height.loc[use_levels] = level_estimate.loc[use_levels]
    final_source.loc[use_levels] = "levels_estimate"

    observed_heights = final_height.dropna()
    average_height = (
        observed_heights.mean() if not observed_heights.empty else DEFAULT_HEIGHT_METERS
    )

    roof_adjustment = roof_heights.copy()
    roof_level_estimate = roof_levels * float(meters_per_level)
    roof_adjustment = roof_adjustment.where(roof_adjustment.notna(), roof_level_estimate)

    use_roof = final_height.isna() & roof_adjustment.notna()
    final_height.loc[use_roof] = average_height + roof_adjustment.loc[use_roof]
    final_source.loc[use_roof] = "roof_adjusted"

    remaining_missing = final_height.isna()
    final_height.loc[remaining_missing] = average_height
    final_source.loc[remaining_missing] = "global_average"

    gdf["height_final_m"] = final_height
    gdf["height_final_source"] = final_source
    return gdf


def _iter_polygons(geom: Polygon | MultiPolygon | GeometryCollection | None) -> Iterator[Polygon]:
    """Yield Polygon parts from any polygonal geometry."""

    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
        return
    if isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            if not part.is_empty:
                yield part
        return
    if isinstance(geom, GeometryCollection):
        for part in geom.geoms:
            if isinstance(part, (Polygon, MultiPolygon)):
                yield from _iter_polygons(part)


def _clean_polygon(poly: Polygon) -> Polygon | None:
    """Return a valid polygon or None when invalid/degenerate."""

    if poly is None or poly.is_empty:
        return None
    if not isinstance(poly, Polygon):
        return None
    if poly.is_valid and poly.area > 0.0:
        return poly

    fixed = poly.buffer(0)
    if isinstance(fixed, Polygon) and fixed.area > 0.0:
        return fixed
    if isinstance(fixed, MultiPolygon):
        # If buffer split the polygon, keep the largest valid component.
        largest = max((part for part in fixed.geoms if part.area > 0.0), default=None, key=lambda p: p.area)
        if largest is not None:
            return largest
    return None


def _signed_area(coords: Sequence[tuple[float, float]]) -> float:
    """Return signed area, positive for counter-clockwise rings."""

    if len(coords) < 3:
        return 0.0
    area = 0.0
    x_prev, y_prev = coords[-1]
    for x_curr, y_curr in coords:
        area += x_prev * y_curr - x_curr * y_prev
        x_prev, y_prev = x_curr, y_curr
    return 0.5 * area


def _triangle_coordinates(poly: Polygon) -> list[list[tuple[float, float]]]:
    """Return CCW triangle coordinate lists covering the polygon area without holes."""

    triangles: list[list[tuple[float, float]]] = []
    for candidate in triangulate(poly):
        if not isinstance(candidate, Polygon) or candidate.is_empty:
            continue
        clipped = candidate.intersection(poly)
        for part in _iter_polygons(clipped):
            if part.area <= 0.0:
                continue
            simple_parts: Iterable[Polygon]
            if len(part.exterior.coords) - 1 > 3:
                simple_parts = triangulate(part)
            else:
                simple_parts = (part,)
            for tri in simple_parts:
                if not isinstance(tri, Polygon) or tri.is_empty:
                    continue
                coords = [(float(x), float(y)) for x, y in tri.exterior.coords[:-1]]
                if len(coords) != 3:
                    continue
                area = _signed_area(coords)
                if area == 0.0:
                    continue
                if area < 0.0:
                    coords = [coords[0], coords[2], coords[1]]
                triangles.append(coords)
    return triangles


def polygon_with_holes_to_surface(poly: Polygon, z0: float = 0.0) -> pv.PolyData:
    """Triangulate a polygon (with holes) into a flat PyVista surface."""

    cleaned = _clean_polygon(poly)
    if cleaned is None:
        raise ValueError("Polygon is empty or invalid.")

    triangles = _triangle_coordinates(cleaned)
    if not triangles:
        raise ValueError("Could not triangulate polygon.")

    xy_to_idx: dict[tuple[float, float], int] = {}
    points: list[tuple[float, float, float]] = []
    faces: list[int] = []

    def _vertex_index(x: float, y: float) -> int:
        key = (x, y)
        idx = xy_to_idx.get(key)
        if idx is None:
            idx = len(points)
            xy_to_idx[key] = idx
            points.append((x, y, float(z0)))
        return idx

    for coords in triangles:
        idxs = [_vertex_index(x, y) for x, y in coords]
        faces.extend([3, idxs[0], idxs[1], idxs[2]])

    surface = pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64))
    surface = surface.clean(tolerance=0.0)
    surface.point_data["base_z"] = np.full(surface.n_points, float(z0), dtype=float)
    return surface


def extrude_polygon_to_shell(poly: Polygon, height: float, z0: float = 0.0) -> pv.PolyData:
    """Extrude a polygon into a watertight LoD1 shell."""

    height = float(height)
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError("Height must be a positive finite number.")

    base = polygon_with_holes_to_surface(poly, z0=z0)
    shell = base.extrude((0.0, 0.0, height), capping=True)
    shell = shell.triangulate().clean().compute_normals(consistent_normals=True)
    if getattr(shell, "n_open_edges", 0) != 0:
        raise ValueError("Extruded shell contains open edges.")
    shell.point_data["height"] = np.full(shell.n_points, height, dtype=float)
    if shell.n_cells > 0:
        shell.cell_data["height"] = np.full(shell.n_cells, height, dtype=float)
    shell.field_data["base_z"] = np.asarray([float(z0)], dtype=float)
    return shell


def _default_height_policy_factory(
    default_height: float = DEFAULT_HEIGHT_METERS,
    meters_per_level: float = DEFAULT_METERS_PER_LEVEL,
) -> Callable[[pd.Series], float]:
    """Create the default height policy callable."""

    priority_columns: tuple[str, ...] = (
        "height_final_m",
        "height",
        "height_m",
    )

    level_columns: tuple[str, ...] = ("levels", "building:levels")

    def _policy(row: pd.Series) -> float:
        for column in priority_columns:
            value = row.get(column)
            if pd.notna(value):
                try:
                    height = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(height) and height > 0.0:
                    return height

        for column in level_columns:
            value = row.get(column)
            if pd.notna(value):
                try:
                    levels = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(levels) and levels > 0.0:
                    return levels * meters_per_level

        return float(default_height)

    return _policy


def _ensure_metric_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project GeoDataFrame to a metric CRS if required."""

    if gdf.crs is None:
        logger.warning("GeoDataFrame has no CRS; projecting to %s.", DEFAULT_METRIC_CRS)
        return gdf.to_crs(DEFAULT_METRIC_CRS)

    crs = CRS.from_user_input(gdf.crs)
    if crs.is_geographic:
        logger.warning("GeoDataFrame CRS is geographic; projecting to %s.", DEFAULT_METRIC_CRS)
        return gdf.to_crs(DEFAULT_METRIC_CRS)
    return gdf


def lod1_buildings_to_meshes(
    gdf: gpd.GeoDataFrame,
    height_policy: Callable[[pd.Series], float] | None = None,
    z0: float = 0.0,
    combine: bool = False,
    default_height: float = DEFAULT_HEIGHT_METERS,
    meters_per_level: float = DEFAULT_METERS_PER_LEVEL,
) -> list[tuple[str, pv.PolyData]] | pv.PolyData:
    """Convert building footprints to LoD1 PyVista meshes."""

    if gdf is None or gdf.empty:
        return pv.PolyData() if combine else []

    work_gdf = _ensure_metric_crs(gdf)
    if "height_final_m" not in work_gdf.columns:
        work_gdf = estimate_building_heights(
            work_gdf,
            meters_per_level=meters_per_level,
        )
    policy = height_policy or _default_height_policy_factory(
        default_height=default_height, meters_per_level=meters_per_level
    )

    outputs: list[tuple[str, pv.PolyData]] = []
    label_to_scalar: dict[str, float] = {}

    for idx, row in work_gdf.iterrows():
        geom = row.geometry
        polygons = list(_iter_polygons(geom))
        if not polygons:
            logger.warning("Skipping feature %s: geometry is empty or not polygonal.", idx)
            continue

        base_label = row.get("building_id", idx)
        base_label = str(base_label) if pd.notna(base_label) else str(idx)

        try:
            height_val = float(policy(row))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Height policy failed for %s (%s); using default.", idx, exc)
            height_val = float(default_height)

        if not np.isfinite(height_val) or height_val <= 0.0:
            logger.warning("Skipping feature %s: invalid height %s.", idx, height_val)
            continue

        for part_idx, polygon in enumerate(polygons):
            cleaned = _clean_polygon(polygon)
            if cleaned is None:
                logger.warning("Skipping polygon part %s:%s due to invalid geometry.", idx, part_idx)
                continue

            label = base_label
            if len(polygons) > 1:
                label = f"{base_label}_part_{part_idx + 1}"

            scalar_id = label_to_scalar.setdefault(label, float(len(label_to_scalar) + 1))
            try:
                mesh = extrude_polygon_to_shell(cleaned, height=height_val, z0=z0)
            except ValueError as exc:
                logger.warning("Skipping polygon part %s: %s", label, exc)
                continue
            if mesh.n_cells > 0:
                mesh.cell_data["building_id"] = np.full(mesh.n_cells, scalar_id, dtype=float)
            outputs.append((label, mesh))

    if combine:
        if not outputs:
            return pv.PolyData()
        scene = pv.merge([mesh for _, mesh in outputs])
        if scene.n_cells > 0 and "building_id" not in scene.cell_data:
            ids = []
            for _, mesh in outputs:
                ids.append(mesh.cell_data.get("building_id", np.full(mesh.n_cells, np.nan)))
            if ids:
                scene.cell_data["building_id"] = np.concatenate(ids)
        return scene

    return outputs


__all__ = [
    "estimate_building_heights",
    "polygon_with_holes_to_surface",
    "extrude_polygon_to_shell",
    "lod1_buildings_to_meshes",
]
