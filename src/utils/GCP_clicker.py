import json

import rasterio
import numpy as np
import cv2

# San Francisco

# frame_path = "1293562080.02321601_sc00113_c1_PAN_i0000000185"
# frame_path = "1293562079.26564479_sc00113_c1_PAN_i0000000150"
# frame_path = "1293562079.69835258_sc00113_c1_PAN_i0000000170"

#Angkor Wat

# frame_path = "1291951336.19337702_sc00103_c1_PAN_i0000000100"
# frame_path = "1291951336.84309149_sc00103_c1_PAN_i0000000130"
# frame_path = "1291951335.11095381_sc00103_c1_PAN_i0000000050"

#Cocabamba

frame_path = "1293376734.34837317_sc00113_c1_PAN_i0000000200"

city = "Cocabamba"

clicked_pixel = (0,0)

VIEW_W, VIEW_H = 200, 200

with rasterio.open(f"../../{city}/l1a_frames/" + frame_path + ".tif") as src:
    img = src.read(1).astype(np.float32)

    img_scaled = (img - img.min()) / (img.max() - img.min()) * 255
    img_scaled = img_scaled.astype(np.uint8)

    vmin = np.percentile(img_scaled, 2)
    vmax = np.percentile(img_scaled, 98)
    img_scaled = np.clip((img_scaled - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

H, W = img_scaled.shape

offset_x, offset_y = 0, 0

def update_view():
    view = img_scaled[offset_y:offset_y+VIEW_H, offset_x:offset_x+VIEW_W]
    view_color = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    # rysuj zaznaczone piksele jeśli są w widoku
    if offset_x <= clicked_pixel[0] < offset_x+VIEW_W and offset_y <= clicked_pixel[1] < offset_y+VIEW_H:
        vx = clicked_pixel[0] - offset_x
        vy = clicked_pixel[1] - offset_y

        for i in range(-3, 3):
            for j in range(-3, 3):
                if 0 <= vx+i < VIEW_W and 0 <= vy+j < VIEW_H:
                    view_color[vy+i, vx+j] = [0, 0, 255]

    cv2.imshow("image", view_color)

def on_trackbar_x(val):
    global offset_x
    offset_x = val
    update_view()

def on_trackbar_y(val):
    global offset_y
    offset_y = val
    update_view()

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global offset_x, offset_y
        global clicked_pixel

        real_x = offset_x + x
        real_y = offset_y + y

        print(f"clicked: ({real_x}, {real_y})")
        clicked_pixel = (real_x, real_y)

        update_view()

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

cv2.createTrackbar("X", "image", 0, W - VIEW_W, on_trackbar_x)
cv2.createTrackbar("Y", "image", 0, H - VIEW_H, on_trackbar_y)

cv2.setMouseCallback("image", click_event)

update_view()

cv2.waitKey(0)

with open(f"../../{city}/own_GCPs/image_position.json", "r") as f:
    realGCPsposition = json.load(f)

if frame_path not in realGCPsposition[frame_path.split("_")[2]]:
    realGCPsposition[frame_path.split("_")[2]][frame_path] = {}
if "GCPs" not in realGCPsposition[frame_path.split("_")[2]][frame_path]:
    realGCPsposition[frame_path.split("_")[2]][frame_path]["GCPs"] = {}

realGCPsposition[frame_path.split("_")[2]][frame_path]["GCPs"]["manual_click"] = {"row": clicked_pixel[1], "col": clicked_pixel[0], "control": 0}

with open(f"../../{city}/own_GCPs/image_position.json", "w") as f:
    json.dump(realGCPsposition, f, indent=2)

cv2.destroyAllWindows()