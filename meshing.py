
import pyvista as pv
import numpy as np
import geopandas as gpd
import dtcc

from typing import Iterable, Optional, Tuple, Sequence, Mapping
from shapely import (LineString,
                    MultiLineString,
                    Point, 
                    Polygon, 
                    MultiPolygon
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import triangulate


def _iter_lines(geometry: BaseGeometry) -> Iterable[LineString]:
    """Yield individual LineStrings from Line/MultiLine geometries."""

    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        for geom in geometry.geoms:
            if geom.is_empty:
                continue
            yield geom


def _iter_polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    """Yield Polygon parts from polygonal geometries."""

    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        for geom in geometry.geoms:
            if geom.is_empty:
                continue
            yield geom


def dtcc_mesh_to_pyvista_mesh(mesh: dtcc.Mesh) -> pv.PolyData:
    points = np.asarray(mesh.vertices, dtype=float)   # (N, 3)
    faces  = np.asarray(mesh.faces, dtype=np.int64)   # (F, 3), triangle indices

    n_faces = faces.shape[0]

    # Build VTK-style faces array: (F, 4) -> [3, i0, i1, i2]
    vtk_faces = np.empty((n_faces, 4), dtype=np.int64)
    vtk_faces[:, 0]  = 3          # 3 vertices per face (triangles)
    vtk_faces[:, 1:] = faces      # copy indices

    faces_flat = vtk_faces.ravel()    # shape (F * 4,)

    return pv.PolyData(points, faces_flat)


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

def _roads_to_polydata(
    roads: gpd.GeoDataFrame,
    z: float = 0.0,
) -> Optional[pv.PolyData]:
    """Convert road edges to a PyVista PolyData made of polylines."""

    if roads is None or roads.empty or "geometry" not in roads:
        return None

    points: list[Tuple[float, float, float]] = []
    lines: list[int] = []
    for geometry in roads.geometry:
        for line in _iter_lines(geometry):
            coords = np.asarray(line.coords)
            if coords.shape[0] < 2:
                continue
            start_idx = len(points)
            for x, y in coords:
                points.append((float(x), float(y), z))
            lines.append(coords.shape[0])
            lines.extend(range(start_idx, start_idx + coords.shape[0]))

    if not points or not lines:
        return None

    points_arr = np.asarray(points, dtype=float)
    lines_arr = np.asarray(lines, dtype=np.int64)
    return pv.PolyData(points_arr, lines=lines_arr)


def _water_to_polydata(
    water: gpd.GeoDataFrame,
    z: float = -0.1,
) -> Optional[pv.PolyData]:
    """Convert waterbody polygons into a single PyVista surface mesh."""

    if water is None or water.empty or "geometry" not in water:
        return None

    surfaces: list[pv.PolyData] = []
    for geometry in water.geometry:
        for polygon in _iter_polygons(geometry):
            try:
                surface = polygon_with_holes_to_surface(polygon, z0=z)
            except ValueError:
                continue
            surfaces.append(surface)

    if not surfaces:
        return None
    if len(surfaces) == 1:
        return surfaces[0]
    merged = pv.merge(surfaces)
    return merged


def _landuse_categories_to_polydata(
    landuse: gpd.GeoDataFrame,
    categories: Iterable[str] = ("GRASS", "FOREST", "FARMLAND"),
    category_column: str = "broad_category",
    base_z: float = 0.0,
    category_z_offsets: Optional[Mapping[str, float]] = None,
) -> dict[str, pv.PolyData]:
    """Triangulate selected landuse polygons into PyVista surfaces.

    Returns a dictionary mapping each requested category to a merged surface mesh.
    """

    if landuse is None or landuse.empty or "geometry" not in landuse:
        return {}
    if category_column not in landuse:
        return {}

    requested = {cat for cat in categories if isinstance(cat, str)}
    if not requested:
        return {}

    subset = landuse[landuse[category_column].isin(requested)]
    if subset.empty:
        return {}

    surfaces_per_category: dict[str, list[pv.PolyData]] = {cat: [] for cat in requested}
    offset_lookup = category_z_offsets or {}

    for _, row in subset.iterrows():
        category = row.get(category_column)
        if category not in requested:
            continue
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue

        z0 = base_z + float(offset_lookup.get(category, 0.0))
        for polygon in _iter_polygons(geometry):
            try:
                surface = polygon_with_holes_to_surface(polygon, z0=z0)
            except ValueError:
                continue
            surfaces_per_category[category].append(surface)

    merged_meshes: dict[str, pv.PolyData] = {}
    for category, surfaces in surfaces_per_category.items():
        if not surfaces:
            continue
        merged_meshes[category] = surfaces[0] if len(surfaces) == 1 else pv.merge(surfaces)
    return merged_meshes
