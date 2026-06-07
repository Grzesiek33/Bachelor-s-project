import csv
import json
from pathlib import Path
from shapely.geometry import Point, Polygon
from shapely import wkt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FrameGCPs = {}
with open("../../landsat/GCPlib.json", "r", encoding="utf-8") as GCPlib:
    with open("../../landsat/GCPlib_ref.json", "r", encoding="utf-8") as GCPlib_ref:
        with open("../../San_francisco/frame_index.csv", "r", encoding="utf-8") as frame_index:
            JGCPlib = json.load(GCPlib)
            JGCPlib_ref = json.load(GCPlib_ref)

            for GCP in JGCPlib:
                GCPlat = float(JGCPlib[GCP]["lat"])
                GCPlon = float(JGCPlib[GCP]["lon"])

                point = Point(GCPlon, GCPlat)

                plt.scatter(GCPlon, GCPlat, c="purple", marker=",", s=1)

            for GCP in JGCPlib_ref:
                GCPlat = float(JGCPlib_ref[GCP]["lat"])
                GCPlon = float(JGCPlib_ref[GCP]["lon"])

                point = Point(GCPlon, GCPlat)

                plt.scatter(GCPlon, GCPlat, c="yellow", marker=",", s=1)

            with open("../../San_francisco/accuearth/accu_earth_GPCs.json", "r", encoding="utf-8") as accu_earth_GCPs:
                Jaccu_earth_GCPs = json.load(accu_earth_GCPs)

            for GCP in Jaccu_earth_GCPs:
                point = wkt.loads(Jaccu_earth_GCPs[GCP]["point"])
                GCPlat = float(point.y)
                GCPlon = float(point.x)

                plt.scatter(GCPlon, GCPlat, c="red", marker=",", s=1)

            polygon = {}
            reader = csv.DictReader(frame_index)
            for row in reader:
                polygon[row["filename"]] = wkt.loads(row["geom"])

            for frame_name, frame_polygon in polygon.items():
                x, y = frame_polygon.exterior.xy
                plt.plot(x, y, c="blue", alpha=0.01)

            sat = mpatches.Patch(color='blue', label='satellite view')
            accu_earth = mpatches.Patch(color='red', label='accuearth GCPs')
            GCPlib_ref = mpatches.Patch(color='yellow', label='GCPlib reference GCPs')
            GCPlib = mpatches.Patch(color='purple', label='GCPlib GCPs')
            plt.legend(handles=[sat, accu_earth, GCPlib_ref, GCPlib], loc="upper right")

            plt.show()

