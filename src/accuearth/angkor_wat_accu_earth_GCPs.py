import geopandas as gpd
import json

from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely import wkt

gdf = gpd.read_file("../../AccuEarth_GCP_archive/World_AccuEarth_GCP.shp")

poly = Polygon([(-123, 36), (-123, 38), (-121, 38), (-121, 36)]) # San Francisco

transformer = Transformer.from_crs(
        "EPSG:4326",  # lat lon height
        "EPSG:4978",  # ECEF
        always_xy=True
    )

result = {
    row["Name"]: {"point": row["geometry"].wkt, "x_ecef": transformer.transform(row["geometry"].x, row["geometry"].y, 0)[0],
                  "y_ecef": transformer.transform(row["geometry"].x, row["geometry"].y, 0)[1],
                  "z_ecef": transformer.transform(row["geometry"].x, row["geometry"].y, 0)[2],
                  "layer": row["layer"]}
    for _, row in gdf.iterrows()
    if row["geometry"] is not None and poly.contains(row["geometry"])
}

with open("../../San_francisco/accuearth/accu_earth_GPCs.json", "w") as f:
    json.dump(result, f, indent=2)



