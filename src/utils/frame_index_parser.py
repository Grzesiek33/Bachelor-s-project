import csv
import json

city = "San_francisco"

def Parse_GSDs(cites = None):

    if cites is None:
        cites = ["San_francisco", "Angkor_wat", "Cocabamba"]

    for city in cites:
        with open(f"../../{city}/own_GCPs/image_position.json", "r", encoding="utf-8") as f:
            with open(f"../../{city}/frame_index.csv", "r", encoding="utf-8") as frame_index:
                frameInfo = json.load(f)

                GSDs = {}

                reader = csv.DictReader(frame_index)
                for row in reader:
                    GSDs[row["filename"]] = float(row["gsd"])

                for frame in frameInfo:
                    frameInfo[frame]["GSD"] = GSDs[frame + ".tif"]


                print(frameInfo)
                with open(f"../../{city}/own_GCPs/image_position.json", "w", encoding="utf-8") as f:
                    json.dump(frameInfo, f, indent=4)

Parse_GSDs()