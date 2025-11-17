import pandas as pd

# -----------------------------------------------------------
# Mapping dictionary: OSM building values → broad categories
# -----------------------------------------------------------
building_category_map = {
    # Sport
    "stadium": "Sport",
    "grandstand": "Sport",
    "sports_centre": "Sport",
    "sports_hall": "Sport",

    # Education
    "university": "Education",
    "school": "Education",
    "kindergarten": "Education",

    # Residential
    "apartments": "Residential",
    "house": "Residential",
    "residential": "Residential",
    "detached": "Residential",
    "terrace": "Residential",
    "dormitory": "Residential",
    "hut": "Residential",

    # Health
    "hospital": "Health",

    # Culture / Leisure
    "museum": "Culture",
    "gallery": "Culture",
    "theatre": "Culture",
    "library": "Culture",

    # Commercial / Service
    "hotel": "Commercial",
    "office": "Commercial",
    "commercial": "Commercial",
    "retail": "Commercial",
    "restaurant": "Commercial",
    "service": "Commercial",
    "toilets": "Commercial",

    # Government / Public
    "civic": "Government",
    "government": "Government",
    "public": "Government",

    # Religious
    "church": "Religious",
    "chapel": "Religious",
    "monastery": "Religious",
    "mausoleum": "Religious",
    "religious": "Religious",

    # Industrial / Storage
    "warehouse": "Industrial",
    "shed": "Industrial",
    "outbuilding": "Industrial",
    "guardhouse": "Industrial",

    # Infrastructure / Structures
    "tower": "Infrastructure",
    "bridge": "Infrastructure",
    "arch": "Infrastructure",
    "roof": "Infrastructure",

    # Military
    "barracks": "Military",
    "military": "Military",

    # Construction / Ruins / Misc
    "construction": "Other",
    "ruins": "Other",
    "yes": "Other",
}

CATEGORY_COLORS = {
    "Residential":        "#1f77b4",
    "Education":          "#ff7f0e",
    "Health":             "#2ca02c",
    "Sport":              "#d62728",
    "Culture/Leisure":    "#9467bd",
    "Commercial/Service": "#8c564b",
    "Government/Public":  "#e377c2",
    "Religious":          "#7f7f7f",
    "Industrial/Storage": "#bcbd22",
    "Infrastructure":     "#17becf",
    "Military":           "#6b6b6b",
    "Other":              "#cccccccd",
}

# -----------------------------------------------------------
# Categorization function
# -----------------------------------------------------------
def categorize_building(value):
    """
    Convert an OSM building value into a broad category.
    - Unknown values → "Other"
    - NaN values → "Other"
    """
    if pd.isna(value):
        return "Other"
    return building_category_map.get(value, "Other")


# -----------------------------------------------------------
# Helper to apply to a GeoDataFrame
# -----------------------------------------------------------
def assign_building_categories(gdf, source_col="building", target_col="classification"):
    """
    Adds a new column with building categories to a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        The input GeoDataFrame.
    source_col : str
        Column containing OSM building values (default: "building").
    target_col : str
        Output column for categories (default: "bldg_category").

    Returns
    -------
    GeoDataFrame
        Same GeoDataFrame with a new category column.
    """
    gdf[target_col] = gdf[source_col].apply(categorize_building)
    return gdf