from __future__ import annotations


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pyvista as pv

import data_ingest
import time
import dtcc
from pathlib import Path

from typing import List

from lod1 import estimate_building_heights
import dtcc
import landuse
import meshing
from buildings_classification import assign_building_categories, CATEGORY_COLORS
from export_plotter import export_plotter

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


def dtcc_buildings_from_geodataframe(buildings_gdf: gpd.GeoDataFrame)-> dict[dtcc.Building]:
    buildings = dict()

    for idx,building_row in buildings_gdf.iterrows():
        building = dtcc.Building()
        footprint_polygon = building_row.geometry
        footprint_surface = dtcc.Surface()
        footprint_surface.from_shapely(footprint_polygon)
        building_id = building_row["uid"]
        height = building_row["height_final_m"]
        footprint_surface.set_z(height)

        building.add_geometry(footprint_surface,dtcc.GeometryType.LOD0)
        building.attributes["name"] = building_row["name"]
        building.attributes["height"] = building_row["height_final_m"]
        buildings[building_id]=building
    return buildings

place = "Thessaloniki, Greece"
side_m = 2000.0
default_meters_per_level = 3.0
config = data_ingest.IngestConfig("all",crs_out="EPSG:2100")
# bbox =  data_ingest.utils.square_from_place(place,side_m)
pamak_xy= (411894.230172,4497388.240600)
white_tower_xy = (410908.457537, 4497522.228564)

bbox = data_ingest.square_from_point(*pamak_xy,
                                     input_crs="EPSG:2100",
                                     side_m= side_m,
                                    )



buildings_gdf = data_ingest.get_buildings(bbox,config)
buildings_gdf = estimate_building_heights(buildings_gdf)
buildings_gdf = assign_building_categories(buildings_gdf)

print(buildings_gdf["classification"])
#Download Road network
roads_nodes, roads_edges, roads_graph  = data_ingest.get_roads(bbox,config)

#Download Landuse
landuse_gdf = data_ingest.get_landuse(bounding_box=bbox, config=config)
landuse_gdf["broad_category"] = landuse_gdf.apply(landuse.classify_row, axis=1)
ax = landuse.plot_landuse_categories(landuse_gdf,opacity=0.3)


roads_edges.plot(ax=ax, color= "black")

buildings_gdf.plot(
    ax=ax,
    facecolor="none",   # transparent fill
    edgecolor="black",  # or e.g. "#444444"
    linewidth=0.5,
    zorder=10           # draw above landuse
)

# buildings_gdf.plot(
#     ax=ax,
#     column="height_final_source",   # your source column (strings)
#     categorical=True,               # treat as categories
#     legend=True,                    # show legend with sources
#     edgecolor="black",              # optional: thin borders
#     linewidth=0.1,
#     zorder=10  
# )

water_gdf = data_ingest.get_water(bbox,config)

# water_gdf.plot(ax=ax,color='blue')

ax.set_title("Buildings colored by height estimation source")
ax.set_axis_off()

plotter = pv.Plotter()
# plt.show()

# buildings= dtcc_buildings_from_geodataframe(buildings_gdf)
# buildings = dtcc.build_lod1_buildings(buildings=buildings,always_use_default_ground=True)
# buildings_multisurfaces: list[dtcc.MultiSurface]= [b.lod1 for _,b in buildings.items()]
# meshes = dtcc.builder.meshing.mesh_multisurfaces(buildings_multisurfaces,clean=False)

# white_tower_mesh = dtcc.load_mesh("/Users/georgespaias/Scratch/big_data_dt_demo/white_tower_surface.vtk")
# for i,mesh in enumerate(meshes):
#         if buildings[i].attributes["name"] == "Λευκός Πύργος":
#             print("Found White Tower!!")
#             centroid = buildings[i].lod0.to_polygon().centroid
#             print(centroid)
#             white_tower_mesh.offset((centroid.x,centroid.y,0.0))
#             meshes[i] = white_tower_mesh
# plotter = pv.Plotter()
# time_begin = time.time()
# for i, mesh in enumerate(meshes):
#     # Get classification for the i-th mesh
#     label = buildings_gdf["classification"].iloc[i]

#     # Pick color (fall back to "Other")
#     color = CATEGORY_COLORS.get(label, CATEGORY_COLORS["Other"])

#     # Convert dtcc mesh → PyVista mesh
#     pv_mesh = dtcc_mesh_to_pyvista_mesh(mesh)

#     # Add to plotter
#     plotter.add_mesh(
#         pv_mesh,
#         show_edges=False,
#         color=color,
#         opacity=1.0,
#     )
# time_end = time.time()
# print("Added meshes in plot in t(s)",time_end-time_begin)
   


roads_mesh = meshing._roads_to_polydata(roads_edges, z=0.0)

if roads_mesh is not None:
    plotter.add_mesh(
            roads_mesh,
            color="gray",
            line_width=2,
            name="road_network",
            render_lines_as_tubes=True,
        )

water_mesh = meshing._water_to_polydata(water_gdf,-0.01)
if water_mesh is not None:
    plotter.add_mesh(
            water_mesh,
            color="#4c77c2",
            opacity=0.5,
            name="water_bodies",
            show_edges=False,
  
        )
    

landuse_meshes = meshing._landuse_categories_to_polydata(
    landuse_gdf,
    categories=("GRASS", "FOREST", "FARMLAND", "RESIDENTIAL", "COMMERCIAL"),
    category_column="broad_category",
    base_z=-0.01,
)

for category, mesh in landuse_meshes.items():
    color = landuse.BROAD_CATEGORY_PALETTE.get(category, "#8dd3ac")
    plotter.add_mesh(
        mesh,
        color=color,
        opacity=0.35,
        name=f"landuse_{category.lower()}",
        show_edges=False,
    )
    

plotter.enable_element_picking()
plotter.show()
