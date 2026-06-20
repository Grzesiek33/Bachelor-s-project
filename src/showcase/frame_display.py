import json
import re
from shapely import wkt
import matplotlib
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from src.optimize.correction_functions import *
from src.utils.PSM_model import *
from src.utils.RFM_model import *

import torch

def show_GCPs_on_frame(frame_path: str, show_projected_GCPs: bool = True, show_optimized_GCPs: bool = True, show_real_GCPs: bool = True,
                       corrected_by: str = "c1", optimized_function = "linear", restricted_to: list = None, exclude: list = None, control_GCPs = None,
                       method: str = 'Nelder-Mead', model = "both", city="San_francisco", cities = None):

    if cities is None:
        cities = ["San_francisco", "Angkor_wat", "Cocabamba"]

    # frame

    with rasterio.open(f"../../{city}/l1a_frames/"+frame_path+".tif") as src:
        img = src.read(1).astype(np.float32)

        img_scaled = (img - img.min()) / (img.max() - img.min()) * 255
        img_scaled = img_scaled.astype(np.uint8)

        profile = src.profile
        profile.update(dtype=rasterio.uint8)

        # Acquire default dots per inch value of matplotlib
        dpi = matplotlib.rcParams['figure.dpi']

        # Determine the figures size in inches to fit your image
        height, width = img.shape
        figsize = width / float(dpi), height / float(dpi)

        plt.figure(figsize=figsize)

        plt.imshow(img_scaled, cmap="gray", origin="upper", vmin=np.percentile(img_scaled,2), vmax=np.percentile(img_scaled,98))

    # data

    with open(f"../../{city}/FrameGCPs.json", "r") as f:
        FrameGCPs = json.load(f)

    GCPs = FrameGCPs[frame_path+".tif"]

    colors = ["red", "green", "blue", "orange", "purple", "cyan", "magenta", "yellow", "brown", "pink"]

    GCPs_colors = {GCP: colors[i % len(colors)] for i, GCP in enumerate(GCPs)}


    with open(f"../../{city}/l1a_frames/"+frame_path+"_pinhole.json", "r") as f:
        FramePSMinfo = json.load(f)

    with open(f"../../{city}/frameRPC.json", "r") as f:
        FrameRFMinfo = json.load(f)[frame_path]

    with open(f"../../{city}/own_GCPs/GCPs.json", "r") as f:
        GCPinfo = json.load(f)

    realGCPsposition = {}

    for ct in cities:
        with open(f"../../{ct}/own_GCPs/image_position.json", "r") as f:
            realGCPsposition[ct] = json.load(f)

    if control_GCPs is None:
        control_GCPs = {}
        for ct in cities:
            control_GCPs[ct] = ""
            for cam in realGCPsposition[ct]:
                for frame in realGCPsposition[ct][cam]:
                    for GCP in realGCPsposition[ct][cam][frame]["GCPs"]:
                        if realGCPsposition[ct][cam][frame]["GCPs"][GCP]["control"] == 1:
                            control_GCPs[ct] += GCP + ", "

    P_projective = torch.tensor(FramePSMinfo["P_projective"], dtype=torch.float64)

    if method == "gradient":

        P_camera = torch.tensor(FramePSMinfo["P_camera"], dtype=torch.float64)
        P_intrinsic = torch.tensor(FramePSMinfo["P_intrinsic"], dtype=torch.float64)

    else:
        P_camera = np.array(FramePSMinfo["P_camera"], dtype=np.float64)
        P_intrinsic = np.array(FramePSMinfo["P_intrinsic"], dtype=np.float64)

    if show_projected_GCPs:
        RFM_model = RFM_torch(FrameRFMinfo["LAT_OFF"], FrameRFMinfo["LAT_SCALE"], FrameRFMinfo["LONG_OFF"],
                        FrameRFMinfo["LONG_SCALE"], FrameRFMinfo["HEIGHT_OFF"], FrameRFMinfo["HEIGHT_SCALE"],
                        FrameRFMinfo["LINE_OFF"], FrameRFMinfo["LINE_SCALE"], FrameRFMinfo["SAMP_OFF"],
                        FrameRFMinfo["SAMP_SCALE"], torch.tensor(FrameRFMinfo["LINE_NUM_COEFFS"], dtype=torch.float64),
                        torch.tensor(FrameRFMinfo["LINE_DEN_COEFFS"], dtype=torch.float64),
                        torch.tensor(FrameRFMinfo["SAMP_NUM_COEFFS"], dtype=torch.float64),
                        torch.tensor(FrameRFMinfo["SAMP_DEN_COEFFS"], dtype=torch.float64))
        for GCP in GCPs:
            meta_data = GCPinfo[GCP]
            x_ecef = float(meta_data["x_ecef"])
            y_ecef = float(meta_data["y_ecef"])
            z_ecef = float(meta_data["z_ecef"])

            B = float(meta_data["lat"])
            L = float(meta_data["lon"])
            H = float(meta_data["alt"])

            im_space = P_projective @ torch.tensor([x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64)

            im_x = im_space[0] / im_space[2]
            im_y = im_space[1] / im_space[2]
            if model == "both" or model == "PSM":
                plt.scatter(im_x, im_y, color=GCPs_colors[GCP], marker="x", label=f"{GCP} PSM", s=100)

            im_y, im_x = RFM_model(B, L, H)
            if model == "both" or model == "RFM":
                plt.scatter(im_x, im_y, color=GCPs_colors[GCP], marker="+", label=f"{GCP} RFM", s=100)

    if show_optimized_GCPs:

        with open(f"../../optimization/" + optimized_function + "_RFM.json", "r") as f:
            optimized_results_RFM = json.load(f)

            if corrected_by[0] == "c":
                params = optimized_results_RFM[corrected_by][method][str(cities)][str(control_GCPs)][
                    ("[]" if exclude is None else "e" + str(exclude)) if restricted_to is None else "r" + str(
                        restricted_to)]
            else:
                params = optimized_results_RFM[frame_path][method][str(cities)][str(control_GCPs)][
                    ("[]" if exclude is None else "e" + str(exclude)) if restricted_to is None else "r" + str(
                        restricted_to)]

        if method == "gradient":
            params = torch.tensor(params, dtype=torch.float64)

            line_num_coeffs = torch.tensor(FrameRFMinfo["LINE_NUM_COEFFS"], dtype=torch.float64)
            line_den_coeffs = torch.tensor(FrameRFMinfo["LINE_DEN_COEFFS"], dtype=torch.float64)
            samp_num_coeffs = torch.tensor(FrameRFMinfo["SAMP_NUM_COEFFS"], dtype=torch.float64)
            samp_den_coeffs = torch.tensor(FrameRFMinfo["SAMP_DEN_COEFFS"], dtype=torch.float64)
            args = torch.cat([line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs])

            correction_function_RFM = globals()[optimized_function](args, no_parameters=80, numpy=False, linear_constraint=1)

            line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs = torch.split(correction_function_RFM(params), [20, 20, 20, 20])

        else:
            params = np.array(params, dtype=np.float64)
            line_num_coeffs = np.array(FrameRFMinfo["LINE_NUM_COEFFS"], dtype=np.float64)
            line_den_coeffs = np.array(FrameRFMinfo["LINE_DEN_COEFFS"], dtype=np.float64)
            samp_num_coeffs = np.array(FrameRFMinfo["SAMP_NUM_COEFFS"], dtype=np.float64)
            samp_den_coeffs = np.array(FrameRFMinfo["SAMP_DEN_COEFFS"], dtype=np.float64)
            args = np.concat([line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs])

            correction_function_RFM = globals()[optimized_function](args, no_parameters=80, numpy=True,
                                                                    linear_constraint=1)

            corrected_parameters = correction_function_RFM(params)

            line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs = np.split(corrected_parameters, [20, 40, 60])

        if method == "gradient":

            RFM_model = RFM_torch(FrameRFMinfo["LAT_OFF"], FrameRFMinfo["LAT_SCALE"], FrameRFMinfo["LONG_OFF"],
                        FrameRFMinfo["LONG_SCALE"], FrameRFMinfo["HEIGHT_OFF"], FrameRFMinfo["HEIGHT_SCALE"],
                        FrameRFMinfo["LINE_OFF"], FrameRFMinfo["LINE_SCALE"], FrameRFMinfo["SAMP_OFF"],
                        FrameRFMinfo["SAMP_SCALE"], line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs)
        else:
            RFM_model = RFM_numpy(FrameRFMinfo["LAT_OFF"], FrameRFMinfo["LAT_SCALE"], FrameRFMinfo["LONG_OFF"],
                        FrameRFMinfo["LONG_SCALE"], FrameRFMinfo["HEIGHT_OFF"], FrameRFMinfo["HEIGHT_SCALE"],
                        FrameRFMinfo["LINE_OFF"], FrameRFMinfo["LINE_SCALE"], FrameRFMinfo["SAMP_OFF"],
                        FrameRFMinfo["SAMP_SCALE"], line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs)

        with open(f"../../optimization/" + optimized_function + "_PSM.json", "r") as f:

            optimized_results_PSM = json.load(f)

            if corrected_by[0] == "c":
                corrected_exterior_rotation = optimized_results_PSM[corrected_by][method][str(cities)][str(control_GCPs)][
                    ("[]" if exclude is None else "e" + str(exclude)) if restricted_to is None else "r" + str(
                        restricted_to)]
            else:
                corrected_exterior_rotation = optimized_results_PSM[frame_path][method][str(cities)][str(control_GCPs)][
                    ("[]" if exclude is None else "e" + str(exclude)) if restricted_to is None else "r" + str(
                        restricted_to)]

        original_exterior_rotation = FramePSMinfo["exterior_orientation"]

        if method == "gradient":
            quaternion = torch.tensor(
                [original_exterior_rotation["qw_ecef"], original_exterior_rotation["qx_ecef"],
                 original_exterior_rotation["qy_ecef"],
                 original_exterior_rotation["qz_ecef"]], dtype=torch.float64)

            sat_position = torch.tensor(
                [original_exterior_rotation["x_ecef_meters"], original_exterior_rotation["y_ecef_meters"],
                 original_exterior_rotation["z_ecef_meters"]], dtype=torch.float64)

            corrected_exterior_rotation = torch.tensor(corrected_exterior_rotation, dtype=torch.float64)

            correction_function = globals()[optimized_function](torch.cat([quaternion, sat_position]), no_parameters=7,
                                                                numpy=False, linear_constraint=1e-4)

            corrected_parameters = correction_function(corrected_exterior_rotation)

            quaternion, sat_position = torch.split(corrected_parameters, [4, 3])

            P_extrinsic = create_extrinsic_torch(quaternion, sat_position)
        else:
            quaternion = np.array(
                [original_exterior_rotation["qw_ecef"], original_exterior_rotation["qx_ecef"],
                 original_exterior_rotation["qy_ecef"],
                 original_exterior_rotation["qz_ecef"]], dtype=np.float64)

            sat_position = np.array(
                [original_exterior_rotation["x_ecef_meters"], original_exterior_rotation["y_ecef_meters"],
                 original_exterior_rotation["z_ecef_meters"]], dtype=np.float64)

            corrected_exterior_rotation = np.array(corrected_exterior_rotation, dtype=np.float64)

            correction_function = globals()[optimized_function](np.concat([quaternion, sat_position]), no_parameters=7,
                                                                numpy=True, linear_constraint=1e-4)

            corrected_parameters = correction_function(corrected_exterior_rotation)

            quaternion, sat_position = np.split(corrected_parameters, [4])

            P_extrinsic = create_extrinsic_numpy(quaternion, sat_position)

        for GCP in GCPs:

                meta_data = GCPinfo[GCP]
                x_ecef = float(meta_data["x_ecef"])
                y_ecef = float(meta_data["y_ecef"])
                z_ecef = float(meta_data["z_ecef"])

                B = float(meta_data["lat"])
                L = float(meta_data["lon"])
                H = float(meta_data["alt"])

                if method == "gradient":
                    im_space = P_camera @ P_intrinsic @ P_extrinsic @ torch.tensor([x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64)
                else:
                    im_space = P_camera @ P_intrinsic @ P_extrinsic @ np.array([x_ecef, y_ecef, z_ecef, 1], dtype=np.float64)
                im_x = im_space[0] / im_space[2]
                im_y = im_space[1] / im_space[2]
                if model == "both" or model == "PSM":
                    plt.scatter(im_x, im_y, color=GCPs_colors[GCP], marker="o", label=f"{GCP} (corrected position PSM)", s=100)

                im_y, im_x = RFM_model(B, L, H)

                if model == "both" or model == "RFM":
                    plt.scatter(im_x, im_y, color=GCPs_colors[GCP], marker="*", label=f"{GCP} (corrected position RFM)", s=100)

    if show_real_GCPs:
        for GCP in realGCPsposition[city][frame_path.split("_")[2]][frame_path]["GCPs"]:
            real_im_x = realGCPsposition[city][frame_path.split("_")[2]][frame_path]["GCPs"][GCP]["col"]
            real_im_y = realGCPsposition[city][frame_path.split("_")[2]][frame_path]["GCPs"][GCP]["row"]

            if realGCPsposition[city][frame_path.split("_")[2]][frame_path]["GCPs"][GCP]["control"]:
                marker = "^"
            else:
                marker = "v"

            plt.scatter(real_im_x, real_im_y, color=GCPs_colors[GCP], marker=marker, label=f"{GCP} (real position) "+("control" if realGCPsposition[city][frame_path.split("_")[2]][frame_path]["GCPs"][GCP]["control"]==1 else "used for correction"), s=100)

    plt.tight_layout()
    plt.subplots_adjust(right=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20)
    plt.title(f"corrected on {corrected_by} using {optimized_function} function with {method} method"+(f" restricted to {restricted_to}" if restricted_to is not None else "")+(f" excluding {exclude}" if exclude is not None else ""), fontsize=20)
    plt.show()

if __name__ == "__main__":

    # show_GCPs_on_frame("1293376734.34837317_sc00113_c1_PAN_i0000000200", show_real_GCPs=False, show_optimized_GCPs=False, show_projected_GCPs=False, city="Cocabamba")

    # show_GCPs_on_frame("1293562079.26564479_sc00113_c1_PAN_i0000000150", method_PSM="gradient", optimized_function="shift")


    show_GCPs_on_frame("1293562080.02321601_sc00113_c1_PAN_i0000000185", optimized_function="shift", show_real_GCPs=True, show_projected_GCPs=True)
    # show_GCPs_on_frame("1293562080.02321601_sc00113_c1_PAN_i0000000185", method="gradient", optimized_function="linear")
    # show_GCPs_on_frame("1293562080.02321601_sc00113_c1_PAN_i0000000185", method_PSM="gradient", optimized_function="quadratic")

    # show_GCPs_on_frame("1293562079.69835258_sc00113_c1_PAN_i0000000170", method_PSM="gradient", optimized_function="quadratic")
    # show_GCPs_on_frame("1293562079.69835258_sc00113_c1_PAN_i0000000170", method_PSM="gradient", optimized_function="linear")

    # show_GCPs_on_frame("1291951336.19337702_sc00103_c1_PAN_i0000000100", method_PSM="gradient", optimized_function="shift", city="Angkor_wat")
    # show_GCPs_on_frame("1291951336.19337702_sc00103_c1_PAN_i0000000100", method_PSM="gradient", optimized_function="linear", city="Angkor_wat")
    # show_GCPs_on_frame("1291951336.19337702_sc00103_c1_PAN_i0000000100", method_PSM="gradient", optimized_function="quadratic", city="Angkor_wat")

    # show_GCPs_on_frame("1293376734.34837317_sc00113_c1_PAN_i0000000200", method_PSM="gradient", optimized_function="shift", city="Cocabamba")
    # show_GCPs_on_frame("1293376734.34837317_sc00113_c1_PAN_i0000000200", method_PSM="gradient", optimized_function="linear", city="Cocabamba")
    # show_GCPs_on_frame("1293376734.34837317_sc00113_c1_PAN_i0000000200", method_PSM="gradient", optimized_function="quadratic", city="Cocabamba")
