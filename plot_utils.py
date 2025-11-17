import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_landuse(
    landuse: gpd.GeoDataFrame,
    area_boundary: gpd.GeoDataFrame | None = None,
    category_column: str = "landuse",
    cmap_name: str = "tab20",
) -> None:
    """Plot landuse polygons using a unique color for each landuse value."""

    if landuse is None or landuse.empty:
        print("No landuse polygons to plot.")
        return

    gdf = landuse.copy()
    fallback_label = "Unknown"
    if category_column in gdf.columns:
        category_values = gdf[category_column]
    else:
        category_values = pd.Series(fallback_label, index=gdf.index, dtype=object)

    normalized = (
        category_values.fillna(fallback_label)
        .astype(str)
        .str.strip()
    )
    normalized = normalized.replace("", fallback_label)

    unique_categories = normalized.drop_duplicates().tolist()
    cmap = plt.cm.get_cmap(cmap_name, max(1, len(unique_categories)))
    color_lookup = {cat: to_hex(cmap(idx)) for idx, cat in enumerate(unique_categories)}
    face_colors = normalized.map(color_lookup).tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    legend_handles: list = []

    boundary_color = "#444444"
    if area_boundary is not None and not area_boundary.empty:
        area_boundary.boundary.plot(ax=ax, color=boundary_color, linewidth=1.0)
        legend_handles.append(
            Line2D([0], [0], color=boundary_color, linewidth=1.2, label="Area boundary")
        )

    gdf.plot(ax=ax, color=face_colors, edgecolor="#222222", linewidth=0.5)

    legend_handles.extend(
        Patch(facecolor=color_lookup[category], edgecolor="#222222", label=category)
        for category in unique_categories
    )

    ax.set_title("Landuse map")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    plt.tight_layout()
    plt.show()



def _plot_landuse_boundaries(
        
    
    landuse: gpd.GeoDataFrame,
    area_boundary: gpd.GeoDataFrame|None,
    nature_color: str = "#2e8b57",
    other_color: str = "#d97706",
) -> None:
    """Plot landuse boundaries, highlighting green/nature polygons."""

    if landuse is None or landuse.empty:
        print("No landuse polygons to plot.")
        return

    def _normalized(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip().str.lower()

    green_landuse_values = {
        "grass",
        "flowerbed",
        "meadow",
        "forest",
        "farmland",
        "greenfield",
        "allotments",
        "village_green",
        "orchard",
        "vineyard",
        "park",
    }
    natural_keywords = {
        "grassland",
        "scrub",
        "wood",
        "heath",
        "fell",
        "bare_rock",
        "sand",
        "shrub",
        "wetland",
    }
    leisure_green_values = {
        "garden",
        "park",
        "golf_course",
        "recreation_ground",
    }

    landuse_values = _normalized(landuse.get("landuse", pd.Series()))
    natural_values = _normalized(landuse.get("natural", pd.Series()))
    leisure_values = _normalized(landuse.get("leisure", pd.Series()))

    nature_mask = (
        landuse_values.isin(green_landuse_values)
        | natural_values.isin(natural_keywords)
        | leisure_values.isin(leisure_green_values)
    )

    nature_landuse = landuse[nature_mask]
    other_landuse = landuse[~nature_mask]

    fig, ax = plt.subplots(figsize=(10, 8))
    if area_boundary is not None and not area_boundary.empty:
        area_boundary.boundary.plot(ax=ax, color="#444444", linewidth=1.0, label="Area boundary")

    if not nature_landuse.empty:
        nature_landuse.boundary.plot(
            ax=ax,
            color=nature_color,
            linewidth=1.2,
            label="Grass/flowerbed/nature landuse",
        )

    if not other_landuse.empty:
        other_landuse.boundary.plot(
            ax=ax,
            color=other_color,
            linewidth=0.9,
            label="Other landuse",
        )

    ax.set_title("Landuse boundaries")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    plt.tight_layout()
    plt.show()
