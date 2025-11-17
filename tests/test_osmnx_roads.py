"""Tests for the osmnx road ingestion helper."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point, Polygon

from osmnx_ingest import IngestConfig
from osmnx_ingest.workflow import get_roads


def test_get_roads_projects_and_enriches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper should add speeds/times, project, and convert to GeoDataFrames."""

    area_geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    cfg = IngestConfig(crs_out="utm", add_speeds_times=True)

    base_graph = nx.MultiDiGraph()
    base_graph.graph["crs"] = "EPSG:4326"
    base_graph.add_node(1)
    base_graph.add_node(2)
    base_graph.add_edge(1, 2, key=0)

    calls = {"speeds": False, "times": False, "project": False}

    def _graph_from_polygon(*args, **kwargs):  # type: ignore[no-untyped-def]
        return base_graph

    def _add_edge_speeds(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        calls["speeds"] = True
        graph.graph["speeds_added"] = True
        return graph

    def _add_edge_travel_times(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        calls["times"] = True
        graph.graph["times_added"] = True
        return graph

    def _project_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        calls["project"] = True
        projected = graph.copy()
        projected.graph["crs"] = "EPSG:32634"
        return projected

    nodes_gdf = gpd.GeoDataFrame(
        {"value": [1, 2]},
        geometry=gpd.GeoSeries(
            [Point(0.0, 0.0), Point(1.0, 1.0)], crs="EPSG:32634"
        ),
        crs="EPSG:32634",
    )
    edges_gdf = gpd.GeoDataFrame(
        {"length": [1.0]},
        geometry=[LineString([(0.0, 0.0), (1.0, 1.0)])],
        crs="EPSG:32634",
    )

    def _graph_to_gdfs(graph: nx.MultiDiGraph, **kwargs):  # type: ignore[no-untyped-def]
        assert graph.graph.get("crs") == "EPSG:32634"
        return nodes_gdf, edges_gdf

    monkeypatch.setattr(
        "osmnx_ingest.workflow.ox.graph_from_polygon", _graph_from_polygon
    )
    monkeypatch.setattr("osmnx_ingest.workflow.ox.add_edge_speeds", _add_edge_speeds)
    monkeypatch.setattr(
        "osmnx_ingest.workflow.ox.add_edge_travel_times", _add_edge_travel_times
    )
    monkeypatch.setattr("osmnx_ingest.workflow.ox.project_graph", _project_graph)
    monkeypatch.setattr("osmnx_ingest.workflow.ox.graph_to_gdfs", _graph_to_gdfs)

    nodes, edges, graph = get_roads(area_geom, cfg)

    assert calls == {"speeds": True, "times": True, "project": True}
    assert nodes is nodes_gdf
    assert edges is edges_gdf
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.graph.get("crs") == "EPSG:32634"


def test_get_roads_returns_empty_layers_for_overpass_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty Overpass responses should yield empty GeoDataFrames instead of errors."""

    class EmptyOverpassResponse(Exception):
        """Match the name used by osmnx for empty responses."""

    def _graph_from_polygon(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise EmptyOverpassResponse()

    monkeypatch.setattr(
        "osmnx_ingest.workflow.ox.graph_from_polygon", _graph_from_polygon
    )

    nodes, edges, graph = get_roads(
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        IngestConfig(),
    )

    assert nodes.empty
    assert edges.empty
    assert nodes.crs and nodes.crs.to_string() == "EPSG:4326"
    assert edges.crs and edges.crs.to_string() == "EPSG:4326"
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 0
