"""Example script that fetches buildings near Paris using the ingest package."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pyvista as pv
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, Point, box
from shapely.geometry.base import BaseGeometry

from lod1 import (
    estimate_building_heights,
    lod1_buildings_to_meshes,
    polygon_with_holes_to_surface,
)

import osmnx_ingest  
import time
import dtcc
from pathlib import Path
import plot_utils
import landuse



METERS_PER_LEVEL = 3.5


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

def _square_from_place(place: str, side_m: float) -> BaseGeometry:
    """Create a square polygon in WGS84 centered around the place centroid."""

    area_gdf = ox.geocode_to_gdf(place)
    if area_gdf.empty:
        raise ValueError(f"Could not resolve '{place}' to a geometry.")

    centroid = area_gdf.geometry.iloc[0].centroid
    centroid_gdf = gpd.GeoSeries([centroid], crs="EPSG:4326")
    utm_crs = centroid_gdf.estimate_utm_crs()
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

def _square_from_point(x: float, y: float, input_crs: str | CRS, side_m: float) -> BaseGeometry:
    """Create a square polygon centered on the point for the requested side length."""

    if side_m <= 0:
        raise ValueError("side_m must be positive to form a bounding box.")

    point_series = gpd.GeoSeries([Point(x, y)], crs=input_crs)
    if point_series.crs is None:
        raise ValueError("input_crs must be provided to construct the bounding box.")

    crs_obj = point_series.crs
    # axis_units = {axis.unit_name.lower() for axis in crs_obj.axis_info if axis.unit_name}
    # if crs_obj.is_projected and any(unit.startswith("metre") or unit.startswith("meter") for unit in axis_units):
    #     metric_series = point_series
    #     metric_crs = point_series.crs
    # else:
    #     wgs_series = point_series.to_crs("EPSG:4326")
    #     utm_crs = wgs_series.estimate_utm_crs()
    #     metric_series = wgs_series.to_crs(utm_crs)
    #     metric_crs = utm_crs
    metric_series = point_series
    half_side = side_m / 2.0
    metric_point = metric_series.iloc[0]
    rect = box(
        metric_point.x - half_side,
        metric_point.y - half_side,
        metric_point.x + half_side,
        metric_point.y + half_side,
    )
    return gpd.GeoSeries([rect], crs=crs_obj).to_crs("EPSG:4326").iloc[0]

def _plot_building_footprints(
    buildings: gpd.GeoDataFrame,
    area_boundary: gpd.GeoDataFrame,
    height_column: str = "height_final_m",
) -> None:
    """Plot building footprints colored by the provided height column."""

    if buildings.empty:
        print("No building footprints to plot.")
        return

    valid = buildings.dropna(subset=[height_column])
    if valid.empty:
        print("No height information available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    if area_boundary is not None and not area_boundary.empty:
        area_boundary.boundary.plot(ax=ax, color="black", linewidth=1.0)

    valid.plot(
        column=height_column,
        ax=ax,
        edgecolor="black",
        linewidth=0.2,
        cmap="viridis",
        legend=True,
    )

    ax.set_title("Building footprints colored by height (m)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


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


def update_unique_csv(series, csv_path, column_name="value"):
    # New unique values from current GeoDataFrame
    new_vals = pd.Series(series.dropna().astype(str).unique(), name=column_name)

    path = Path(csv_path)

    if path.exists():
        # Load existing values
        old_vals = pd.read_csv(path)[column_name].astype(str).unique()
        combined = sorted(set(old_vals).union(new_vals))
    else:
        combined = sorted(set(new_vals))

    # Save back to CSV (overwrite with deduplicated + sorted values)
    pd.DataFrame({column_name: combined}).to_csv(path, index=False)

def main() -> None:
    place = "Thessaloniki, Greece"
    side_m = 2000.0
    # polygon = _square_from_place(place, side_m)
    #PAMAK
    pamak_xy= (411894.230172,4497388.240600)
    white_tower_xy = (410908.457537, 4497522.228564)
    skyline_xy = (411427.654092, 4497509.488063)
    # polygon = _square_from_point(*pamak_xy,input_crs="EPSG:2100", side_m=side_m)
    polygon = _square_from_place(place,side_m)
    
    cfg = osmnx_ingest.IngestConfig(height_policy="levels_x_3m", crs_out="utm")
    
    osmnx_ingest.workflow.apply_osmnx_settings(cfg)
    area_geom = osmnx_ingest.fetch_area(place=None, polygon=polygon, bbox=None)

    buildings_gdf = osmnx_ingest.get_buildings(area_geom, cfg)

    # buildings_gdf.set_crs("EPSG:2100",allow_override=True)
    print(buildings_gdf)
    
    print(
        f"Fetched {len(buildings_gdf)} buildings_gdf within a {side_m} m square around {place}."
    )
    buildings_gdf = estimate_building_heights(buildings_gdf, meters_per_level=METERS_PER_LEVEL)
    buildings_gdf["geometry"] = buildings_gdf.geometry.simplify(tolerance=0.5, preserve_topology=True)
    buildings_gdf["height"] = buildings_gdf["height_final_m"]
    if "building:levels" in buildings_gdf.columns:
        buildings_gdf["levels"] = buildings_gdf["building:levels"]
    if "osm_id" in buildings_gdf.columns:
        buildings_gdf["building_id"] = buildings_gdf["osm_id"].astype(str)
    else:
        buildings_gdf["building_id"] = buildings_gdf.index.astype(str)
    print(
        buildings_gdf[[
            "osm_id",
            "height_m",
            "building:levels",
            "height_final_m",
            "height_final_source",
        ]]
        .head()
    )

    # buildings = []
    # for idx,building_row in buildings_gdf.iterrows():
    #     building = dtcc.Building()
    #     footprint_polygon = building_row.geometry
    #     footprint_surface = dtcc.Surface()
    #     footprint_surface.from_shapely(footprint_polygon)
        
    #     print(building_row["name"])
    #     height = building_row["height_final_m"]
    #     footprint_surface.set_z(height)

    #     building.add_geometry(footprint_surface,dtcc.GeometryType.LOD0)
    #     building.attributes["name"] = building_row["name"]
    #     building.attributes["height"] = building_row["height_final_m"]
    #     buildings.append(building)

    


    # # city.add_buildings(buildings)

    # buildings = dtcc.build_lod1_buildings(buildings=buildings,always_use_default_ground=True)
    

    # buildings_multisurfaces= [b.lod1 for b in buildings]
    
    # meshes = dtcc.builder.meshing.mesh_multisurfaces(buildings_multisurfaces)
    # white_tower_mesh = dtcc.load_mesh("/Users/georgespaias/Scratch/big_data_dt_demo/white_tower_surface.vtk")

    # for i,mesh in enumerate(meshes):
    #     if buildings[i].attributes["name"] == "Λευκός Πύργος":
    #         centroid = buildings[i].lod0.to_polygon().centroid
    #         print(centroid)
    #         white_tower_mesh.offset((centroid.x,centroid.y,0.0))
    #         meshes[i] = white_tower_mesh
       
    # for building in buildings:
    #     print(building.attributes["name"],building.mesh)
   
    # plotter = pv.Plotter()
    # time_begin = time.time()
    # for mesh in meshes:
        
    #     # mesh = building.mesh
    #     print("Adding:",mesh)
    #     pv_mesh = dtcc_mesh_to_pyvista_mesh(mesh)
    #     plotter.add_mesh(
    #         pv_mesh,
    #         show_edges=False,
    #         opacity=1.0,
    #     )
    # time_end = time.time()
    # print("Added meshes in plot in t(s)",time_end-time_begin)
   

    # roads_nodes, roads_edges, roads_graph  = osmnx_ingest.get_roads(area_geom,cfg)
    # roads_mesh = _roads_to_polydata(roads_edges, z=0.0)

    # if roads_mesh is not None:
    #     plotter.add_mesh(
    #         roads_mesh,
    #         color="gray",
    #         line_width=2,
    #         name="road_network",
    #         render_lines_as_tubes=True,
    #     )

    # water = osmnx_ingest.get_water(area_geom, cfg)

    # if not water.empty:   
    #       ax = plt.gca()
    #       water.plot(ax=ax, edgecolor="black", facecolor="#0a4c70")
    #       # area_boundary.plot(ax=ax, facecolor="none", edgecolor="#e31a1c", linewidth=2)
          
    # plt.show()
    # water_mesh = _water_to_polydata(water,-0.01)
    # if water_mesh is not None:
    #     plotter.add_mesh(
    #         water_mesh,
    #         color="#4c77c2",
    #         opacity=0.5,
    #         name="water_bodies",
    #         show_edges=False,
  
    #     )

    # plotter.show()

    landuse_gdf = osmnx_ingest.get_landuse(area_geom,cfg)
    
    # plot_utils._plot_landuse_boundaries(landuse,None)
    # plot_utils.plot_landuse(landuse,None)
    
    landuse_gdf["broad_category"] = landuse_gdf.apply(landuse.classify_row, axis=1)

    ax = landuse.plot_landuse_categories(landuse_gdf)
    buildings_gdf.plot(
    ax=ax,
    facecolor="none",   # transparent fill
    edgecolor="black",  # or e.g. "#444444"
    linewidth=0.5,
    zorder=10           # draw above landuse
)
    
    water_gdf = osmnx_ingest.get_water(area_geom,cfg)
    
  
    
    print(buildings_gdf.crs, landuse_gdf.crs,water_gdf.crs)

    # buildings_gdf.plot(ax=ax,color='gray')
   
    buildings_gdf.plot(ax=ax, edgecolor='black',facecolor='none')
    plt.show()
    
if __name__ == "__main__":
    main()