import json
import re
from shapely import wkt
import matplotlib
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sympy import latex

from src.optimize.correction_functions import *
from src.utils.PSM_model import *
from src.utils.RFM_model import *

import torch

def makeAccuracyPlot():

    with open(f"../../San_francisco/own_GCPs/GCPs.json", "r") as f:
        GCPinfo = json.load(f)

    with open(f"../../optimization/shift_PSM.json", "r") as f:
        optim = json.load(f)

    with open(f"../../San_francisco/own_GCPs/image_position.json", "r") as f:
        realGCPsposition = json.load(f)

    error = []
    error_m = []

    for i in range(1, 11):
        tab = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", ""]
        del tab[i-1]
        params = optim["c1"]["gradient"]["['San_francisco', 'Angkor_wat', 'Cochabamba']"][f"{{'San_francisco': '{', '.join(tab)}', 'Angkor_wat': '1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ', 'Cochabamba': '1, 2, 3, '}}"]["[]"]

        error.append([])
        error_m.append([])

        for frame in realGCPsposition["c1"]:

            with open(f"../../San_francisco/l1a_frames/" + frame + "_pinhole.json", "r") as f:
                FramePSMinfo = json.load(f)

            P_camera = torch.tensor(FramePSMinfo["P_camera"], dtype=torch.float64)
            P_intrinsic = torch.tensor(FramePSMinfo["P_intrinsic"], dtype=torch.float64)

            for GCP in realGCPsposition[frame.split("_")[2]][frame]["GCPs"]:
                if(GCP != str(i)):

                    meta_data = GCPinfo[GCP]
                    x_ecef = float(meta_data["x_ecef"])
                    y_ecef = float(meta_data["y_ecef"])
                    z_ecef = float(meta_data["z_ecef"])

                    real_im_x = realGCPsposition[frame.split("_")[2]][frame]["GCPs"][GCP]["col"]
                    real_im_y = realGCPsposition[frame.split("_")[2]][frame]["GCPs"][GCP]["row"]

                    original_exterior_rotation = FramePSMinfo["exterior_orientation"]

                    quaternion = torch.tensor(
                        [original_exterior_rotation["qw_ecef"], original_exterior_rotation["qx_ecef"],
                         original_exterior_rotation["qy_ecef"],
                         original_exterior_rotation["qz_ecef"]], dtype=torch.float64)

                    sat_position = torch.tensor(
                        [original_exterior_rotation["x_ecef_meters"], original_exterior_rotation["y_ecef_meters"],
                         original_exterior_rotation["z_ecef_meters"]], dtype=torch.float64)

                    quaternion, sat_position = globals()["shift_PSM"](
                        [torch.tensor(p, dtype=torch.float64) for p in params], quaternion,
                        sat_position)

                    P_extrinsic_corrected = create_extrinsic_torch(quaternion, sat_position)

                    im_space = P_camera @ P_intrinsic @ P_extrinsic_corrected @ torch.tensor(
                        [x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64)

                    pred_im_x_PSM = im_space[0] / im_space[2]
                    pred_im_y_PSM = im_space[1] / im_space[2]

                    error[i-1].append(np.sqrt((pred_im_x_PSM.item() - real_im_x) ** 2 + (pred_im_y_PSM.item() - real_im_y) ** 2))
                    error_m[i-1].append(error[i-1][-1] * FramePSMinfo["GSD"])
        error_m[i-1] = np.mean(error_m[i-1])

    plt.plot(range(1, 11), error_m, marker="o")
    plt.xlabel("GCP used for optimization")
    plt.ylabel("Mean error (meters)")
    plt.title("Accuracy of PSM correction for a chosen GCP")

    plt.xticks(range(1, 11))
    plt.grid()
    plt.show()

if __name__ == "__main__":
    makeAccuracyPlot()