
from __future__ import annotations

import math
import re
from typing import Dict, Tuple, Optional

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from shapely.geometry import Polygon,box
import networkx as nx
from dtcc import Bounds

from .config import IngestConfig
from .utils  import apply_osmnx_settings,empty_generic_layer


def get_roads(
    bounding_box: Bounds | Polygon, config: IngestConfig
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, nx.MultiDiGraph]:
    """Fetch the road network (nodes/edges + graph) for the requested area."""

    apply_osmnx_settings(config)

    if isinstance(bounding_box,Bounds):
      bounding_box = box(minx=bounding_box.xmin,
                miny=bounding_box.ymin,
                maxx=bounding_box.xmax,
                maxy=bounding_box.ymax
		)
    try:
        graph = ox.graph_from_polygon(
            bounding_box,
            network_type=config.roadnetwork_type,
            simplify=config.simplify_roads,
        )
    except Exception as exc:  # pragma: no cover - pass back unexpected failures
        if exc.__class__.__name__ == "EmptyOverpassResponse":
            empty_roads = empty_generic_layer("EPSG:4326")
            return (
                empty_roads.copy(),
                empty_roads.copy(),
                nx.MultiDiGraph(),
            )
        raise

    if graph is None or graph.number_of_nodes() == 0:
        base_crs = None
        if graph is not None:
            base_crs = graph.graph.get("crs")
        empty_roads = empty_generic_layer(base_crs)
        return (empty_roads.copy(), empty_roads.copy(), nx.MultiDiGraph())

    if config.add_speeds_times and graph.number_of_edges() > 0:
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)

    if config.crs_out == "utm":
        graph = ox.project_graph(graph)

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(
        graph,
        nodes=True,
        edges=True,
        node_geometry=True,
        fill_edge_geometry=True,
    )
    if config.crs_out is None:
        config.crs_out = "EPSG:4326"
    return nodes_gdf.to_crs( config.crs_out), edges_gdf.to_crs( config.crs_out), graph