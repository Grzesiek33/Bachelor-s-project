import json
from pyproj import Transformer

fields = [
"scene_id",
"scene_active",
"scene_update_date",
"phase",
"version",
"path",
"row",
"gls",
"lat",
"lon",
"elevation",
"gcp_type",

"chip_id",
"chip_active",
"chip_update_date",
"chip_version",
"chip_path",
"chip_row",

"ref_line",
"ref_sample",

"proj_x",
"proj_y",

"pixel_size_x",
"pixel_size_y",

"size_lines",
"size_samples",

"sensor",
"selection_method",
"moravec_rank",

"projection",
"utm_zone",
"chip_type",
"acquisition_date",
"data_type"
]

data = {}

with open("../../landsat/GCPLib_ref.txt") as f:
    lines = f.readlines()

start = False

for line in lines:
    line = line.strip()

    if line == "BEGIN":
        start = True
        continue

    if not start:
        continue

    if line.isdigit():
        continue

    parts = line.split()

    record = dict(zip(fields, parts))

    chip_id = record["chip_id"]

    data[chip_id] = record


    transformer = Transformer.from_crs(
        "EPSG:4326",  # lat lon height
        "EPSG:4978",  # ECEF
        always_xy=True
    )

    x, y, z = transformer.transform(data[chip_id]["lon"], data[chip_id]["lat"], data[chip_id]["elevation"])

    data[chip_id]["x_ecef"] = x
    data[chip_id]["y_ecef"] = y
    data[chip_id]["z_ecef"] = z

with open("../../landsat/GCPlib_ref.json", "w") as f:
    json.dump(data,f,indent=2)