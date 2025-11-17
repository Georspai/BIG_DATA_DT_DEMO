import dtcc
import numpy as np
import pyvista as pv
import geopandas as gpd

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


def dtcc_buildings_from_geodataframe(buildings_gdf: gpd.GeoDataFrame)-> dict[str,dtcc.Building]:
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