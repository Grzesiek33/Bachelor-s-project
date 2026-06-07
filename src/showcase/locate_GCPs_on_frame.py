import csv
import json
from pathlib import Path
from shapely.geometry import Point, Polygon
from shapely import wkt
import numpy as np

city = "Cocabamba"

folder = Path(f"../../{city}/l1a_frames")

FrameGCPs = {}
with open(f"../../{city}/own_GCPs/GCPs.json", "r", encoding="utf-8") as GCPlib:
    with open(f"../../{city}/frame_index.csv", "r", encoding="utf-8") as frame_index:
        JGCPlib = json.load(GCPlib)

        polygon = {}
        reader = csv.DictReader(frame_index)
        for row in reader:
            polygon[row["filename"]] = wkt.loads(row["geom"])

        for frame_name in polygon.keys():
            FrameGCPs[frame_name] = []

        for GCP in JGCPlib:

            point = Point(JGCPlib[GCP]["lat"], JGCPlib[GCP]["lon"])

            for frame_name, frame_polygon in polygon.items():

                if frame_polygon.contains(point):
                    print(f"GCP {GCP} is located within frame {frame_name}")
                    FrameGCPs[frame_name].append(GCP)


        print(FrameGCPs)
        with open(f"../../{city}/FrameGCPs.json", "w", encoding="utf-8") as f:
            json.dump(FrameGCPs, f, indent=4)