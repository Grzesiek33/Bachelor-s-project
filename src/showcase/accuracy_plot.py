import json
import os.path
from itertools import combinations

from shapely import wkt
import matplotlib
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sympy import latex

from src.optimize.correction_functions import *
from src.utils.PSM_model import *
from src.utils.RFM_model import *
from src.utils.cities import *
import torch


def accuracy_plot(no_eval_GCPs, optimized_function = linear, method: str = 'gradient', model ="PSM", city="San_francisco", correction_for="c1", device=torch.device("cpu"), eval_on_trained = False, correction_function_parameters=None):

    assert model in ["PSM", "RFM"], "Model must be 'PSM' or 'RFM'"

    if correction_function_parameters is None:
        if model == "PSM":
            correction_function_parameters = {"linear_constraint": 1e-4, "quadratic_constraint": 1e-12, "no_parameters": 7}
        elif model == "RFM":
            correction_function_parameters = {"linear_constraint": 1, "quadratic_constraint": 1, "no_parameters": 80}

    GCPinfo = {}
    realGCPsposition = {}

    assert os.path.exists(f"../../{city}/own_GCPs/GCPs.json"), f"no available GCP data for {city}"

    with open(f"../../{city}/own_GCPs/GCPs.json", "r") as f:
        GCPinfo[city] = json.load(f)

    assert os.path.exists(f"../../{city}/own_GCPs/image_position.json"), f"no available GCP image position data for {city}"

    with open(f"../../San_francisco/own_GCPs/image_position.json", "r") as f:
        realGCPsposition = json.load(f)

    assert os.path.exists(f"../../optimization/{optimized_function.__name__}_{model}.json"), f"no available optimization data for {model} and {optimized_function.__name__} model"

    with open(f"../../optimization/{optimized_function.__name__}_{model}.json", "r") as f:
        optim = json.load(f)

    allGCPs = {}

    for ct in supported_cities:
        with open(f"../../{ct}/own_GCPs/image_position.json", "r") as f:
            image_position = json.load(f)

        allGCPs[ct] = []
        for frame in image_position:
            for GCP in image_position[frame]["GCPs"]:
                allGCPs[ct].append(GCP)

    error = []
    error_m = []
    GCPs = []

    for i, comb in enumerate(combinations(allGCPs[city], no_eval_GCPs)):

        params = torch.tensor(optim[method][str(correction_function_parameters)][str({city: list(comb)})], dtype=torch.float64, device=device)

        error.append([])
        error_m.append([])
        GCPs.append(str(list(comb)))

        if model == "RFM":
            with open(f"../../{city}/frameRPC.json", "r") as f:
                frameInfo = json.load(f)

        for frame in realGCPsposition:

            if not (correction_for == "all" or (correction_for[0] == "c" and frame.split("_")[2] == correction_for) or (frame in correction_for)):
                continue

            if model == "PSM":
                with open(f"../../San_francisco/l1a_frames/" + frame + "_pinhole.json", "r") as f:
                    FramePSMinfo = json.load(f)

                P_camera = torch.tensor(FramePSMinfo["P_camera"], dtype=torch.float64, device=device)
                P_intrinsic = torch.tensor(FramePSMinfo["P_intrinsic"], dtype=torch.float64, device=device)

                exterior_rotation = FramePSMinfo["exterior_orientation"]

                quaternion = torch.tensor(
                    [exterior_rotation["qw_ecef"], exterior_rotation["qx_ecef"],
                     exterior_rotation["qy_ecef"],
                     exterior_rotation["qz_ecef"]], dtype=torch.float64, device=device)

                sat_position = torch.tensor(
                    [exterior_rotation["x_ecef_meters"], exterior_rotation["y_ecef_meters"],
                     exterior_rotation["z_ecef_meters"]], dtype=torch.float64, device=device)

                correction_function = optimized_function(torch.cat([quaternion, sat_position]), numpy=False, **correction_function_parameters, device=device)

                corrected_parameters = correction_function(params)

                q, sat_pos = torch.split(corrected_parameters, [4, 3])

                model_eval = PSM(P_camera, P_intrinsic, q, sat_pos, numpy=False, device=device)

            else:
                Line_num_coeffs = torch.tensor(frameInfo[frame][f"LINE_NUM_COEFFS"], dtype=torch.float64, device=device)
                Line_den_coeffs = torch.tensor(frameInfo[frame][f"LINE_DEN_COEFFS"], dtype=torch.float64, device=device)
                Samp_num_coeffs = torch.tensor(frameInfo[frame][f"SAMP_NUM_COEFFS"], dtype=torch.float64, device=device)
                Samp_den_coeffs = torch.tensor(frameInfo[frame][f"SAMP_DEN_COEFFS"], dtype=torch.float64, device=device)

                line_off = frameInfo[frame]["LINE_OFF"]
                samp_off = frameInfo[frame]["SAMP_OFF"]
                lat_off = frameInfo[frame]["LAT_OFF"]
                long_off = frameInfo[frame]["LONG_OFF"]
                height_off = frameInfo[frame]["HEIGHT_OFF"]
                line_scale = frameInfo[frame]["LINE_SCALE"]
                samp_scale = frameInfo[frame]["SAMP_SCALE"]
                lat_scale = frameInfo[frame]["LAT_SCALE"]
                long_scale = frameInfo[frame]["LONG_SCALE"]
                height_scale = frameInfo[frame]["HEIGHT_SCALE"]

                correction_function = optimized_function(torch.cat([Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs]),
                    numpy=False, **correction_function_parameters, device=device)

                corrected_parameters = correction_function(params)

                Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs = torch.split(
                    corrected_parameters, [20, 20, 20, 20])

                model_eval = RFM(lat_off, lat_scale, long_off, long_scale, height_off, height_scale,
                            line_off, line_scale, samp_off, samp_scale, Line_num_coeffs,
                            Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs, numpy=False, device=device)

            for GCP in list(set(realGCPsposition[frame]["GCPs"]) - set(comb)) if not eval_on_trained else realGCPsposition[frame]["GCPs"]:

                meta_data = GCPinfo[city][GCP]
                if model == "PSM":
                    x_ecef = float(meta_data["x_ecef"])
                    y_ecef = float(meta_data["y_ecef"])
                    z_ecef = float(meta_data["z_ecef"])
                    im_x, im_y = model_eval(x_ecef, y_ecef, z_ecef)
                else:
                    B = float(meta_data["lat"])
                    L = float(meta_data["lon"])
                    H = float(meta_data["alt"])
                    im_x, im_y = model_eval(B, L, H)

                real_im_x = realGCPsposition[frame]["GCPs"][GCP]["col"]
                real_im_y = realGCPsposition[frame]["GCPs"][GCP]["row"]

                error[i].append(np.sqrt((im_x.item() - real_im_x) ** 2 + (im_y.item() - real_im_y) ** 2))
                error_m[i].append(error[i][-1] * realGCPsposition[frame]["GSD"])
        error_m[-1] = np.mean(error_m[-1])

    plt.plot(GCPs, error_m, marker="o")
    plt.xlabel("GCPs used for optimization")
    plt.ylabel("Mean error (meters)")
    plt.title(f"Accuracy for {model} using {optimized_function.__name__} correction")

    plt.grid()
    plt.show()

if __name__ == "__main__":
    accuracy_plot(no_eval_GCPs=1, optimized_function=shift, model="RFM")