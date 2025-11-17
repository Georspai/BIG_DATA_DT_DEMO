import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BROAD_CATEGORY_PALETTE = {
    "GRASS":        "#74c476",
    "FOREST":       "#238b45",
    "FARMLAND":     "#c2e699",
    "CONSTRUCTION": "#b15928",
    "INDUSTRIAL":   "#737373",
    "RESIDENTIAL":  "#1f78b4",
    "COMMERCIAL":   "#e31a1c",
    "RELIGIOUS":    "#6a51a3",
    "MILITARY":     "#307303FF",
    "UNKNOWN":      "#bdbdbd",
}

LANDUSE_TO_CATEGORY = {
    "allotments":           "FARMLAND",      # community food-growing
    "apiary":               "FARMLAND",      # beekeeping, agricultural use
    "basin":                "UNKNOWN",       # water infrastructure, not in your list
    "brownfield":           "CONSTRUCTION",  # previously developed, often re-development
    "cemetery":             "UNKNOWN",       # could be its own class; default UNKNOWN
    "civic_admin":          "COMMERCIAL",    # admin/civic service area
    "commercial":           "COMMERCIAL",
    "construction":         "CONSTRUCTION",
    "education":            "COMMERCIAL",    # schools/universities ~ service/institution
    "farmland":             "FARMLAND",
    "farmyard":             "FARMLAND",
    "flowerbed":            "GRASS",         # ornamental vegetation; closest is GRASS
    "forest":               "FOREST",
    "garages":              "RESIDENTIAL",   # often accessory to housing
    "grass":                "GRASS",
    "greenery":             "GRASS",         # decorative green areas
    "greenhouse_horticulture": "FARMLAND",
    "industrial":           "INDUSTRIAL",
    "meadow":               "GRASS",
    "military":             "MILITARY",
    "orchard":              "FARMLAND",
    "railway":              "UNKNOWN",       # transport infra, not in your categories
    "recreation_ground":    "GRASS",         # open recreational grass area
    "religious":            "RESIDENTIAL",
    "residential":          "RESIDENTIAL",
    "retail":               "COMMERCIAL",
    "sc":                   "UNKNOWN",       # unclear abbreviation
    "shrubs":               "GRASS",         # closest to vegetation; use GRASS
    "village_green":        "GRASS",
    "vineyard":             "FARMLAND",
}


LEISURE_TO_CATEGORY = {
    "common":           "GRASS",        # open grassy common
    "garden":           "GRASS",        # residential/ornamental greenery
    "horse_riding":     "UNKNOWN",      # facility, not pure landcover
    "nature_reserve":   "FOREST",       # often forest/woodland; could also be FARMLAND/GRASS
    "park":             "GRASS",        # urban park, mostly grass/trees
    "pitch":            "GRASS",        # sports fields, grass/artificial turf
    "playground":       "RESIDENTIAL",  # typically in/near residential areas
    "sports_centre":    "COMMERCIAL",   # facility-type use
    "track":            "UNKNOWN",      # running/athletics track, ambiguous
}


NATURAL_TO_CATEGORY = {
    "bare_rock":    "UNKNOWN",  # rocky areas; you have no ROCK category
    "grass":        "GRASS",
    "grassland":    "GRASS",
    "heath":        "GRASS",    # low shrub/grass mixture
    "sand":         "UNKNOWN",  # beaches/dunes; no SAND category
    "scrub":        "GRASS",    # low woody, shrubby vegetation
    "shrubbery":    "GRASS",
    "wood":         "FOREST",
}

def classify_row(row):
    # 1) landuse
    lu = row.get("landuse")
    if isinstance(lu, str) and lu in LANDUSE_TO_CATEGORY:
        return LANDUSE_TO_CATEGORY[lu]

    # 2) leisure
    le = row.get("leisure")
    if isinstance(le, str) and le in LEISURE_TO_CATEGORY:
        return LEISURE_TO_CATEGORY[le]

    # 3) natural
    nat = row.get("natural")
    if isinstance(nat, str) and nat in NATURAL_TO_CATEGORY:
        return NATURAL_TO_CATEGORY[nat]

    # fallback
    return "UNKNOWN"



def plot_landuse_categories(gdf, category_col="broad_category",
                            palette=BROAD_CATEGORY_PALETTE,
                            ax=None, legend=True,opacity=0.7, edgecolor="black"):
    """
    Plot polygons in `gdf` colored by `category_col` using `palette`.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain polygon geometries and a categorical column.
    category_col : str
        Column with broad classes (e.g. 'broad_category').
    palette : dict
        Mapping category -> hex color.
    ax : matplotlib Axes, optional
        Existing axes. If None, creates a new figure + axes.
    legend : bool
        Whether to draw a legend.
    edgecolor : str
        Color of polygon borders.
    """
    # pick colors per row; default to UNKNOWN color if missing
    colors = gdf[category_col].map(palette).fillna(palette.get("UNKNOWN", "#bdbdbd"))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(ax=ax, color=colors, edgecolor=edgecolor,alpha=opacity , linewidth=0.2)

    ax.set_axis_off()

    if legend:
        # only legend for categories actually present
        present_cats = sorted(gdf[category_col].dropna().unique())
        patches = [
            mpatches.Patch(color=palette.get(cat, "#bdbdbd"), label=cat)
            for cat in present_cats
        ]
        ax.legend(handles=patches, title="Landuse", loc="upper right")

    return ax