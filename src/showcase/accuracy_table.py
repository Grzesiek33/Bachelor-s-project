import json
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

def makeAccuracyTable(corrected_by: str = "c1", optimized_function = "linear", cities = None, control_GCPs: dict = None,
                      restricted_to: list = None, exclude: list = None, method: str = 'Nelder-Mead', city="San_francisco"):

    if cities is None:
        cities = ["San_francisco", "Angkor_wat", "Cochabamba"]

    with open(f"../../{city}/own_GCPs/GCPs.json", "r") as f:
        GCPinfo = json.load(f)

    control_GCPs_ammount=0
    trained_on_GCPs_ammount=0

    original_prediction_distances_PSM = 0
    original_prediction_distances_RFM = 0
    original_prediction_distances_PSM_m = 0
    original_prediction_distances_RFM_m = 0

    trained_on_distance_PSM=0
    control_distance_PSM=0
    trained_on_distance_PSM_m=0
    control_distance_PSM_m=0

    trained_on_distance_RFM=0
    control_distance_RFM=0
    trained_on_distance_RFM_m=0
    control_distance_RFM_m=0

    realGCPsposition = {}

    for ct in cities:
        with open(f"../../{ct}/own_GCPs/image_position.json", "r") as f:
            realGCPsposition[ct] = json.load(f)

    if control_GCPs is None:
        control_GCPs = {}
        for ct in cities:
            control_GCPs[ct] = 0
            for frame in realGCPsposition[ct]:
                for GCP in realGCPsposition[ct][frame]["GCPs"]:
                    if realGCPsposition[ct][frame]["GCPs"][GCP]["control"] == 1:
                        control_GCPs[ct] += 1

    for frame in realGCPsposition[city][corrected_by]:

        with open(f"../../{city}/l1a_frames/"+frame+"_pinhole.json", "r") as f:
            FramePSMinfo = json.load(f)

        with open(f"../../{city}/frameRPC.json", "r") as f:
            FrameRFMinfo = json.load(f)[frame]

        P_projective = torch.tensor(FramePSMinfo["P_projective"], dtype=torch.float64)
        P_camera = torch.tensor(FramePSMinfo["P_camera"], dtype=torch.float64)
        P_intrinsic = torch.tensor(FramePSMinfo["P_intrinsic"], dtype=torch.float64)

        RFM_model = RFM_torch(FrameRFMinfo["LAT_OFF"], FrameRFMinfo["LAT_SCALE"], FrameRFMinfo["LONG_OFF"],
                        FrameRFMinfo["LONG_SCALE"], FrameRFMinfo["HEIGHT_OFF"], FrameRFMinfo["HEIGHT_SCALE"],
                        FrameRFMinfo["LINE_OFF"], FrameRFMinfo["LINE_SCALE"], FrameRFMinfo["SAMP_OFF"],
                        FrameRFMinfo["SAMP_SCALE"], torch.tensor(FrameRFMinfo["LINE_NUM_COEFFS"], dtype=torch.float64),
                        torch.tensor(FrameRFMinfo["LINE_DEN_COEFFS"], dtype=torch.float64),
                        torch.tensor(FrameRFMinfo["SAMP_NUM_COEFFS"], dtype=torch.float64),
                        torch.tensor(FrameRFMinfo["SAMP_DEN_COEFFS"], dtype=torch.float64))

        for GCP in realGCPsposition[city][frame.split("_")[2]][frame]["GCPs"]:
            meta_data = GCPinfo[GCP]
            x_ecef = float(meta_data["x_ecef"])
            y_ecef = float(meta_data["y_ecef"])
            z_ecef = float(meta_data["z_ecef"])

            lon = float(meta_data["lon"])
            lat = float(meta_data["lat"])
            alt = float(meta_data["alt"])

            real_im_x = realGCPsposition[city][frame.split("_")[2]][frame]["GCPs"][GCP]["col"]
            real_im_y = realGCPsposition[city][frame.split("_")[2]][frame]["GCPs"][GCP]["row"]

            pred_im_x_RFM = RFM_model(lon, lat, alt)[1]
            pred_im_y_RFM = RFM_model(lon, lat, alt)[0]

            im_space = P_projective @ torch.tensor([x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64)

            pred_im_x_PSM = im_space[0] / im_space[2]
            pred_im_y_PSM = im_space[1] / im_space[2]

            original_prediction_distances_RFM += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item()
            original_prediction_distances_PSM += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item()

            original_prediction_distances_RFM_m += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item() * realGCPsposition[city][frame.split("_")[2]][frame]["GSD"]
            original_prediction_distances_PSM_m += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item() * realGCPsposition[city][frame.split("_")[2]][frame]["GSD"]

            line_num_coeffs = torch.tensor(FrameRFMinfo["LINE_NUM_COEFFS"], dtype=torch.float64)
            line_den_coeffs = torch.tensor(FrameRFMinfo["LINE_DEN_COEFFS"], dtype=torch.float64)
            samp_num_coeffs = torch.tensor(FrameRFMinfo["SAMP_NUM_COEFFS"], dtype=torch.float64)
            samp_den_coeffs = torch.tensor(FrameRFMinfo["SAMP_DEN_COEFFS"], dtype=torch.float64)

            with open(f"../../optimization/" + optimized_function + "_RFM.json", "r") as f:
                optimized_results_RFM = json.load(f)

                if corrected_by[0] == "c":
                    params = optimized_results_RFM[corrected_by][method][str(cities)][str(control_GCPs)][
                        ("[]" if exclude is None else "e" + str(exclude)) if restricted_to is None else "r" + str(
                            restricted_to)]
                else:
                    params = optimized_results_RFM[frame][method][str(cities)][str(control_GCPs)][
                        ("[]" if exclude is None else "e" + str(exclude)) if restricted_to is None else "r" + str(
                            restricted_to)]

                params = [torch.tensor(p, dtype=torch.float64) for p in params]

            line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs = globals()[optimized_function + "_RFM"](
                params, line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs)

            RFM_model_corrected = RFM_torch(FrameRFMinfo["LAT_OFF"], FrameRFMinfo["LAT_SCALE"], FrameRFMinfo["LONG_OFF"],
                            FrameRFMinfo["LONG_SCALE"], FrameRFMinfo["HEIGHT_OFF"], FrameRFMinfo["HEIGHT_SCALE"],
                            FrameRFMinfo["LINE_OFF"], FrameRFMinfo["LINE_SCALE"], FrameRFMinfo["SAMP_OFF"],
                            FrameRFMinfo["SAMP_SCALE"], line_num_coeffs, line_den_coeffs, samp_num_coeffs,
                            samp_den_coeffs)

            with open(f"../../optimization/" + optimized_function + "_PSM.json", "r") as f:

                optimized_results_PSM = json.load(f)

                if corrected_by[0] == "c":
                    corrected_exterior_rotation = optimized_results_PSM[corrected_by][method][str(cities)][str(control_GCPs)][
                        ("[]" if exclude is None else "e" + str(exclude)) if restricted_to is None else "r" + str(
                            restricted_to)]
                else:
                    corrected_exterior_rotation = optimized_results_PSM[frame][method][str(cities)][str(control_GCPs)][
                        ("[]" if exclude is None else "e" + str(exclude)) if restricted_to is None else "r" + str(
                            restricted_to)]

            original_exterior_rotation = FramePSMinfo["exterior_orientation"]

            quaternion = torch.tensor(
                [original_exterior_rotation["qw_ecef"], original_exterior_rotation["qx_ecef"],
                 original_exterior_rotation["qy_ecef"],
                 original_exterior_rotation["qz_ecef"]], dtype=torch.float64)

            sat_position = torch.tensor(
                [original_exterior_rotation["x_ecef_meters"], original_exterior_rotation["y_ecef_meters"],
                 original_exterior_rotation["z_ecef_meters"]], dtype=torch.float64)

            quaternion, sat_position = globals()[optimized_function + "_PSM"](
                [torch.tensor(p, dtype=torch.float64) for p in corrected_exterior_rotation], quaternion, sat_position)

            P_extrinsic_corrected = create_extrinsic(quaternion, sat_position, numpy=False)

            pred_im_x_RFM = RFM_model_corrected(lon, lat, alt)[1]
            pred_im_y_RFM = RFM_model_corrected(lon, lat, alt)[0]

            im_space = P_camera @ P_intrinsic @ P_extrinsic_corrected @ torch.tensor([x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64)

            pred_im_x_PSM = im_space[0] / im_space[2]
            pred_im_y_PSM = im_space[1] / im_space[2]

            if realGCPsposition[city][frame.split("_")[2]][frame]["GCPs"][GCP]["control"]:
                control_distance_RFM += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item()
                control_distance_PSM += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item()

                control_distance_RFM_m += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item() * realGCPsposition[city][frame.split("_")[2]][frame]["GSD"]
                control_distance_PSM_m += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item() * realGCPsposition[city][frame.split("_")[2]][frame]["GSD"]

                control_GCPs_ammount += 1
            else:
                trained_on_distance_RFM += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item()
                trained_on_distance_PSM += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item()

                trained_on_distance_RFM_m += torch.sqrt((pred_im_x_RFM - real_im_x)**2 + (pred_im_y_RFM - real_im_y)**2).item() * realGCPsposition[city][frame.split("_")[2]][frame]["GSD"]
                trained_on_distance_PSM_m += torch.sqrt((pred_im_x_PSM - real_im_x)**2 + (pred_im_y_PSM - real_im_y)**2).item() * realGCPsposition[city][frame.split("_")[2]][frame]["GSD"]
                trained_on_GCPs_ammount += 1

    data = {
        "Metric": ["Original", "Trained on", "Control"],
        "RFM_px": [
            original_prediction_distances_RFM / (control_GCPs_ammount + trained_on_GCPs_ammount),
            trained_on_distance_RFM / trained_on_GCPs_ammount if trained_on_GCPs_ammount > 0 else 0,
            control_distance_RFM / control_GCPs_ammount if control_GCPs_ammount > 0 else 0

        ],
        "PSM_px": [
            original_prediction_distances_PSM / (control_GCPs_ammount + trained_on_GCPs_ammount),
            trained_on_distance_PSM / trained_on_GCPs_ammount if trained_on_GCPs_ammount > 0 else 0,
            control_distance_PSM / control_GCPs_ammount if control_GCPs_ammount > 0 else 0
        ],
        "RFM_m": [
            original_prediction_distances_RFM_m / (control_GCPs_ammount + trained_on_GCPs_ammount),
            trained_on_distance_RFM_m / trained_on_GCPs_ammount if trained_on_GCPs_ammount > 0 else 0,
            control_distance_RFM_m / control_GCPs_ammount if control_GCPs_ammount > 0 else 0

        ],
        "PSM_m": [
            original_prediction_distances_PSM_m / (control_GCPs_ammount + trained_on_GCPs_ammount),
            trained_on_distance_PSM_m / trained_on_GCPs_ammount if trained_on_GCPs_ammount > 0 else 0,
            control_distance_PSM_m / control_GCPs_ammount if control_GCPs_ammount > 0 else 0
        ]
    }

    # Wygeneruj tabelę w px

    latex_table_px = r"\begin{tabular}{|c|c|c|}" + "\n"
    latex_table_px += r"\hline" + "\n"
    latex_table_px += r"Metric & RFM [px] & PSM [px] \\" + "\n"
    latex_table_px += r"\hline" + "\n"

    for i, metric in enumerate(data["Metric"]):

        rfm_val = f"{data['RFM_px'][i]:.6f}"
        psm_val = f"{data['PSM_px'][i]:.6f}"

        if i == 1:
            if trained_on_GCPs_ammount == 0:
                rfm_val = "N/A"
                psm_val = "N/A"
        elif i == 2:
            if control_GCPs_ammount == 0:
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
            if trained_on_GCPs_ammount == 0:
                rfm_val = "N/A"
                psm_val = "N/A"
        elif i == 2:
            if control_GCPs_ammount == 0:
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

    if(city=="San_francisco"):
        print(r"\textbf{San Francisco}" + "\n")
    elif(city=="Angkor_wat"):
        print(r"\textbf{Angkor Wat}" + "\n")
    elif(city=="Cochabamba"):
        print(r"\textbf{Cochabamba}" + "\n")

    print(latex_output)


if __name__ == "__main__":

    # print(r"\subsection{Correction: shift}" + "\n")
    #
    # makeAccuracyTable(corrected_by="c1", optimized_function="shift", method="gradient", city="San_francisco")
    # # makeAccuracyTable(corrected_by="c1", optimized_function="shift", method="gradient", city="Angkor_wat")
    # makeAccuracyTable(corrected_by="c1", optimized_function="shift", method="gradient", city="Cochabamba")

    # print(r"\newpage" + "\n")
    # print(r"\subsection{Correction: linear}" + "\n")
    #
    # makeAccuracyTable(corrected_by="c1", optimized_function="linear", method="gradient", city="San_francisco")
    # # makeAccuracyTable(corrected_by="c1", optimized_function="linear", method="gradient", city="Angkor_wat")
    # makeAccuracyTable(corrected_by="c1", optimized_function="linear", method="gradient", city="Cochabamba")
    #
    # # print(r"\newpage" + "\n")
    print(r"\subsection{Correction: quadratic}" + "\n")

    makeAccuracyTable(corrected_by="c1", optimized_function="quadratic", method="gradient", city="San_francisco")
    # makeAccuracyTable(corrected_by="c1", optimized_function="quadratic", method="gradient", city="Angkor_wat")
    makeAccuracyTable(corrected_by="c1", optimized_function="quadratic", method="gradient", city="Cochabamba")

    # print(r"\newpage" + "\n")