import json

from pyproj import Transformer

city = "Cocabamba"

transformer = Transformer.from_crs(
        "EPSG:4326",  # lat lon height
        "EPSG:4978",  # ECEF
        always_xy=True
    )

with open(f"../../{city}/own_GCPs/GCPs.json") as f:
    GCPs = json.load(f)

for gcp in GCPs:
    lat = float(GCPs[gcp]["lat"])
    lon = float(GCPs[gcp]["lon"])
    alt = float(GCPs[gcp]["alt"])

    x_ecef, y_ecef, z_ecef = transformer.transform(lat, lon, alt)
    GCPs[gcp]["x_ecef"] = x_ecef
    GCPs[gcp]["y_ecef"] = y_ecef
    GCPs[gcp]["z_ecef"] = z_ecef

with open(f"../../{city}/own_GCPs/GCPs.json", "w") as f:
    json.dump(GCPs, f, indent=4)

