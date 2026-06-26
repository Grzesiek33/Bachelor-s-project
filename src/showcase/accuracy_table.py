import json
import os

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

from src.utils.cities import supported_cities


def accuracy_table(optimized_function ="shift", cities = None, train_GCPs: dict = None, method: str = 'Nelder-Mead', correction_for="c1", device=torch.device("cpu"), correction_function_parameters_PSM=None, correction_function_parameters_RFM=None):

    if cities is None:
        cities = supported_cities

    GCPinfo = {}
    realGCPsposition = {}

    for city in cities:
        with open(f"../../{city}/own_GCPs/GCPs.json", "r") as f:
            GCPinfo[city] = json.load(f)

        with open(f"../../{city}/own_GCPs/image_position.json", "r") as f:
            realGCPsposition[city] = json.load(f)

    if train_GCPs is None:
        train_GCPs = {}

        for ct in cities:

            train_GCPs[ct] = []
            for frame in realGCPsposition[ct]:
                for GCP in realGCPsposition[ct][frame]["GCPs"]:
                    train_GCPs[ct].append(GCP)

    if correction_function_parameters_PSM is None:
        correction_function_parameters_PSM = {"linear_constraint": 1e-4, "quadratic_constraint": 1e-12, "no_parameters": 7}
    if correction_function_parameters_RFM is None:
        correction_function_parameters_RFM = {"linear_constraint": 1, "quadratic_constraint": 1, "no_parameters": 80}

    control_GCPs_ammount = {}
    trained_on_GCPs_ammount = {}

    original_prediction_distance_PSM = {}
    original_prediction_distance_RFM = {}
    original_prediction_distance_PSM_m = {}
    original_prediction_distance_RFM_m = {}

    trained_on_distance_PSM = {}
    control_distance_PSM = {}
    trained_on_distance_PSM_m = {}
    control_distance_PSM_m = {}

    trained_on_distance_RFM = {}
    control_distance_RFM = {}
    trained_on_distance_RFM_m = {}
    control_distance_RFM_m = {}

    for city in cities:

        control_GCPs_ammount[city] = 0
        trained_on_GCPs_ammount[city] = 0

        original_prediction_distance_PSM[city] = 0
        original_prediction_distance_RFM[city] = 0
        original_prediction_distance_PSM_m[city] = 0
        original_prediction_distance_RFM_m[city] = 0

        trained_on_distance_PSM[city] = 0
        control_distance_PSM[city] = 0
        trained_on_distance_PSM_m[city] = 0
        control_distance_PSM_m[city] = 0

        trained_on_distance_RFM[city] = 0
        control_distance_RFM[city] = 0
        trained_on_distance_RFM_m[city] = 0
        control_distance_RFM_m[city] = 0

        with open(f"../../{city}/frameRPC.json", "r") as f:
            FrameRFMinfo = json.load(f)

        for frame in realGCPsposition[city]:

            if not (correction_for == "all" or (correction_for[0] == "c" and frame.split("_")[2] == correction_for) or (frame in correction_for)):
                continue

            with open(f"../../{city}/l1a_frames/"+frame+"_pinhole.json", "r") as f:
                FramePSMinfo = json.load(f)

            P_projective = torch.tensor(FramePSMinfo["P_projective"], dtype=torch.float64, device=device)
            P_camera = torch.tensor(FramePSMinfo["P_camera"], dtype=torch.float64, device=device)
            P_intrinsic = torch.tensor(FramePSMinfo["P_intrinsic"], dtype=torch.float64, device=device)

            line_off = FrameRFMinfo[frame]["LINE_OFF"]
            samp_off = FrameRFMinfo[frame]["SAMP_OFF"]
            lat_off = FrameRFMinfo[frame]["LAT_OFF"]
            long_off = FrameRFMinfo[frame]["LONG_OFF"]
            height_off = FrameRFMinfo[frame]["HEIGHT_OFF"]
            line_scale = FrameRFMinfo[frame]["LINE_SCALE"]
            samp_scale = FrameRFMinfo[frame]["SAMP_SCALE"]
            lat_scale = FrameRFMinfo[frame]["LAT_SCALE"]
            long_scale = FrameRFMinfo[frame]["LONG_SCALE"]
            height_scale = FrameRFMinfo[frame]["HEIGHT_SCALE"]

            line_num_coeffs = torch.tensor(FrameRFMinfo[frame]["LINE_NUM_COEFFS"], dtype=torch.float64, device=device)
            line_den_coeffs = torch.tensor(FrameRFMinfo[frame]["LINE_DEN_COEFFS"], dtype=torch.float64, device=device)
            samp_num_coeffs = torch.tensor(FrameRFMinfo[frame]["SAMP_NUM_COEFFS"], dtype=torch.float64, device=device)
            samp_den_coeffs = torch.tensor(FrameRFMinfo[frame]["SAMP_DEN_COEFFS"], dtype=torch.float64, device=device)

            model_RFM = RFM(lat_off, lat_scale, long_off, long_scale, height_off, height_scale,
                        line_off, line_scale, samp_off, samp_scale, line_num_coeffs,
                        line_den_coeffs, samp_num_coeffs, samp_den_coeffs, numpy=False, device=device)

            for GCP in realGCPsposition[city][frame]["GCPs"]:
                meta_data = GCPinfo[city][GCP]
                x_ecef = float(meta_data["x_ecef"])
                y_ecef = float(meta_data["y_ecef"])
                z_ecef = float(meta_data["z_ecef"])

                lon = float(meta_data["lon"])
                lat = float(meta_data["lat"])
                alt = float(meta_data["alt"])

                real_im_x = realGCPsposition[city][frame]["GCPs"][GCP]["col"]
                real_im_y = realGCPsposition[city][frame]["GCPs"][GCP]["row"]

                pred_im_x_RFM, pred_im_y_RFM = model_RFM(lat, lon, alt)

                im_space = P_projective @ torch.tensor([x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64, device=device)

                pred_im_x_PSM = im_space[0] / im_space[2]
                pred_im_y_PSM = im_space[1] / im_space[2]

                original_prediction_distance_RFM[city] += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item()
                original_prediction_distance_PSM[city] += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item()

                original_prediction_distance_RFM_m[city] += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item() * realGCPsposition[city][frame]["GSD"]
                original_prediction_distance_PSM_m[city] += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item() * realGCPsposition[city][frame]["GSD"]

                assert os.path.exists(
                    f"../../optimization/" + optimized_function + "_RFM.json"), f"Optimized RFM parameters for function {optimized_function} do not exist."
                with open(f"../../optimization/" + optimized_function + "_RFM.json", "r") as f:
                    optimized_results_RFM = json.load(f)

                assert method in optimized_results_RFM, f"Method {method} not found in optimized RFM results."
                assert str(correction_function_parameters_RFM) in optimized_results_RFM[
                    method], f"Correction function parameters {correction_function_parameters_RFM} not found in optimized RFM results for method {method}."
                assert str(train_GCPs) in optimized_results_RFM[method][
                    str(correction_function_parameters_RFM)], f"Train GCPs {train_GCPs} not found in optimized RFM results for method {method}."

                args = torch.cat([line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs])

                params = optimized_results_RFM[method][str(correction_function_parameters_RFM)][str(train_GCPs)]

                params = torch.tensor(params, dtype=torch.float64, device=device)

                correction_function_RFM = globals()[optimized_function](args, numpy=False,
                                                                        device=device,
                                                                        **correction_function_parameters_RFM)

                line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs = torch.split(
                    correction_function_RFM(params), [20, 20, 20, 20])

                RFM_model_corrected = RFM(lat_off, lat_scale, long_off, long_scale, height_off, height_scale,
                        line_off, line_scale, samp_off, samp_scale, line_num_coeffs,
                        line_den_coeffs, samp_num_coeffs, samp_den_coeffs, numpy=False, device=device)

                assert os.path.exists(
                    f"../../optimization/" + optimized_function + "_PSM.json"), f"Optimized PSM parameters for function {optimized_function} do not exist."

                with open(f"../../optimization/" + optimized_function + "_PSM.json", "r") as f:
                    optimized_results_PSM = json.load(f)

                assert method in optimized_results_PSM, f"Method {method} not found in optimized PSM results."
                assert str(correction_function_parameters_PSM) in optimized_results_PSM[
                    method], f"Correction function parameters {correction_function_parameters_PSM} not found in optimized PSM results for method {method}."
                assert str(train_GCPs) in optimized_results_PSM[method][
                    str(correction_function_parameters_PSM)], f"Train GCPs {train_GCPs} not found in optimized PSM results for method {method}."

                params = optimized_results_PSM[method][str(correction_function_parameters_PSM)][str(train_GCPs)]
                params = torch.tensor(params, dtype=torch.float64,
                                                           device=device)

                original_exterior_rotation = FramePSMinfo["exterior_orientation"]

                quaternion = torch.tensor(
                    [original_exterior_rotation["qw_ecef"], original_exterior_rotation["qx_ecef"],
                     original_exterior_rotation["qy_ecef"],
                     original_exterior_rotation["qz_ecef"]], dtype=torch.float64)

                sat_position = torch.tensor(
                    [original_exterior_rotation["x_ecef_meters"], original_exterior_rotation["y_ecef_meters"],
                     original_exterior_rotation["z_ecef_meters"]], dtype=torch.float64)

                correction_function = globals()[optimized_function](torch.cat([quaternion, sat_position]),
                                                                    numpy=False, **correction_function_parameters_PSM,
                                                                    device=device)

                corrected_parameters = correction_function(params)

                quaternion, sat_position = torch.split(corrected_parameters, [4, 3])

                P_extrinsic_corrected = create_extrinsic(quaternion, sat_position, numpy=False, device=device)

                pred_im_x_RFM, pred_im_y_RFM = RFM_model_corrected(lat, lon, alt)

                im_space = P_camera @ P_intrinsic @ P_extrinsic_corrected @ torch.tensor([x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64)

                pred_im_x_PSM = im_space[0] / im_space[2]
                pred_im_y_PSM = im_space[1] / im_space[2]

                if city not in train_GCPs or GCP not in train_GCPs[city]:
                    control_distance_RFM[city] += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item()
                    control_distance_PSM[city] += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item()

                    control_distance_RFM_m[city] += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item() * realGCPsposition[city][frame]["GSD"]
                    control_distance_PSM_m[city] += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item() * realGCPsposition[city][frame]["GSD"]

                    control_GCPs_ammount[city] += 1
                else:
                    trained_on_distance_RFM[city] += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item()
                    trained_on_distance_PSM[city] += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item()

                    trained_on_distance_RFM_m[city] += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item() * realGCPsposition[city][frame]["GSD"]
                    trained_on_distance_PSM_m[city] += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item() * realGCPsposition[city][frame]["GSD"]
                    trained_on_GCPs_ammount[city] += 1
    for city in cities:
        data = {
            "Metric": ["Original", "Trained on", "Control"],
            "RFM_px": [
                original_prediction_distance_RFM[city] / (control_GCPs_ammount[city] + trained_on_GCPs_ammount[city]),
                trained_on_distance_RFM[city] / trained_on_GCPs_ammount[city] if trained_on_GCPs_ammount[city] > 0 else 0,
                control_distance_RFM[city] / control_GCPs_ammount[city] if control_GCPs_ammount[city] > 0 else 0

            ],
            "PSM_px": [
                original_prediction_distance_PSM[city] / (control_GCPs_ammount[city] + trained_on_GCPs_ammount[city]),
                trained_on_distance_PSM[city] / trained_on_GCPs_ammount[city] if trained_on_GCPs_ammount[city] > 0 else 0,
                control_distance_PSM[city] / control_GCPs_ammount[city] if control_GCPs_ammount[city] > 0 else 0
            ],
            "RFM_m": [
                original_prediction_distance_RFM_m[city] / (control_GCPs_ammount[city] + trained_on_GCPs_ammount[city]),
                trained_on_distance_RFM_m[city] / trained_on_GCPs_ammount[city] if trained_on_GCPs_ammount[city] > 0 else 0,
                control_distance_RFM_m[city] / control_GCPs_ammount[city] if control_GCPs_ammount[city] > 0 else 0

            ],
            "PSM_m": [
                original_prediction_distance_PSM_m[city] / (control_GCPs_ammount[city] + trained_on_GCPs_ammount[city]),
                trained_on_distance_PSM_m[city] / trained_on_GCPs_ammount[city] if trained_on_GCPs_ammount[city] > 0 else 0,
                control_distance_PSM_m[city] / control_GCPs_ammount[city] if control_GCPs_ammount[city] > 0 else 0
            ]
        }
        print(f"\\textbf{{{city}}}" + "\n")
        latex_table_px = r"\begin{tabular}{|c|c|c|}" + "\n"
        latex_table_px += r"\hline" + "\n"
        latex_table_px += r"Metric & RFM [px] & PSM [px] \\" + "\n"
        latex_table_px += r"\hline" + "\n"

        for i, metric in enumerate(data["Metric"]):

            rfm_val = f"{data['RFM_px'][i]:.6f}"
            psm_val = f"{data['PSM_px'][i]:.6f}"

            if i == 1:
                if trained_on_GCPs_ammount[city] == 0:
                    rfm_val = "N/A"
                    psm_val = "N/A"
            elif i == 2:
                if control_GCPs_ammount[city] == 0:
                    rfm_val = "N/A"
                    psm_val = "N/A"

            latex_table_px += f"{metric} & {rfm_val} & {psm_val} \\\\\n"

        latex_table_px += r"\hline" + "\n"
        latex_table_px += r"\end{tabular}"

        latex_table_m = r"\begin{tabular}{|c|c|c|}" + "\n"
        latex_table_m += r"\hline" + "\n"
        latex_table_m += r"Metric & RFM [m] & PSM [m] \\" + "\n"
        latex_table_m += r"\hline" + "\n"

        for i, metric in enumerate(data["Metric"]):
            rfm_val = f"{data['RFM_m'][i]:.6f}"
            psm_val = f"{data['PSM_m'][i]:.6f}"

            if i == 1:
                if trained_on_GCPs_ammount[city] == 0:
                    rfm_val = "N/A"
                    psm_val = "N/A"
            elif i == 2:
                if control_GCPs_ammount[city] == 0:
                    rfm_val = "N/A"
                    psm_val = "N/A"

            latex_table_m += f"{metric} & {rfm_val} & {psm_val} \\\\\n"

        latex_table_m += r"\hline" + "\n"
        latex_table_m += r"\end{tabular}"

        latex_output = r"\begin{center}" + "\n"
        latex_output += r"\begin{minipage}{0.45\textwidth}" + "\n"
        latex_output += r"\centering" + "\n"
        latex_output += latex_table_px + "\n"
        latex_output += r"\end{minipage}\hfill" + "\n"
        latex_output += r"\begin{minipage}{0.45\textwidth}" + "\n"
        latex_output += r"\centering" + "\n"
        latex_output += latex_table_m + "\n"
        latex_output += r"\end{minipage}" + "\n"
        latex_output += r"\end{center}" + "\n"

        print(latex_output + "\n")


if __name__ == "__main__":

    accuracy_table(method="gradient", train_GCPs={"San_francisco": ["1"]})