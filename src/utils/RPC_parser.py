import json
from pathlib import Path

fields = [
"LINE_OFF",
"SAMP_OFF",
"LAT_OFF",
"LONG_OFF",
"HEIGHT_OFF",
"LINE_SCALE",
"SAMP_SCALE",
"LAT_SCALE",
"LONG_SCALE",
"HEIGHT_SCALE"
]

city = "Cochabamba"

folder = Path(rf"../../{city}/l1a_frames")

frameRPC = {}

for path in folder.iterdir():
    if path.suffix == ".TXT":
        with open(path, "r") as f:
            lines = f.readlines()
            rpc_data = {}
            for i in range(10):
                rpc_data[fields[i]] = float(lines[i].split()[1])
            coeffs_list = []
            for i in range(10, 90):
                coeffs_list.append(float(lines[i].split()[1]))
            rpc_data["LINE_NUM_COEFFS"] = coeffs_list[:20]
            rpc_data["LINE_DEN_COEFFS"] = coeffs_list[20:40]
            rpc_data["SAMP_NUM_COEFFS"] = coeffs_list[40:60]
            rpc_data["SAMP_DEN_COEFFS"] = coeffs_list[60:80]


            frameRPC[path.stem[:-4]] = rpc_data

with open(f"../../{city}/frameRPC.json", "w") as f:
    json.dump(frameRPC, f, indent=4)
