import json
import os.path

from shapely import wkt
import matplotlib
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from matplotlib import colors as mcolors
from src.optimize.correction_functions import *
from src.utils.PSM_model import *
from src.utils.RFM_model import *

import torch

from src.utils.cities import supported_cities


def show_GCPs_on_frame(frame_path: str, show_projected_GCPs: bool = True, show_optimized_GCPs: bool = True, show_real_GCPs: bool = True,  optimized_function = linear, train_GCPs = None,
                       method: str = 'gradient', model = "both", city="San_francisco", colors = None, device=torch.device("cpu"), correction_function_parameters_PSM = None, correction_function_parameters_RFM = None):
    """
    Display ground control points (GCPs) on a single image frame, comparing projected,
    optimized and real image positions.

    This function loads a single-band TIFF frame and overlays GCP positions computed
    from the physical sensor model (PSM), the rational function model (RFM),
    their optimized/corrected variants, and the real annotated image positions.
    It is intended for visual inspection of projection/correction quality across
    different optimization methods and correction functions.

    Parameters
    ----------
    frame_path : str
        Base filename (without extension) of the frame inside "../../{city}/l1a_frames/".
        Example: "1293562080.02321601_sc00113_c1_PAN_i0000000185".
    show_projected_GCPs : bool, optional
        If True, plot GCP positions projected by the original PSM and RFM models.
        Default True.
    show_optimized_GCPs : bool, optional
        If True, plot GCP positions after applying optimized correction parameters
        (loaded from the `../../optimization/` JSON files). Default True.
    show_real_GCPs : bool, optional
        If True, plot the real GCP image positions from
        "../../{city}/own_GCPs/image_position.json". Default True.
    optimized_function : callable, optional
        Correction function used to optimize the camera model.
        Default linear.
    train_GCPs : dict or None, optional
        Mapping of city -> list of GCP identifiers used during optimization (the
        "training" GCPs). If None, the function will build a train list from all
        supported cities. Default None.
    method : str, optional
        Optimization method key used to find results in optimization JSON files
        (e.g. "gradient", "Nelder-Mead"). Default 'gradient'.
    model : {'PSM', 'RFM', 'both'}, optional
        Which model(s) to evaluate and plot: only PSM, only RFM, or both.
        Default "both".
    city : str, optional
        City containing the frame to display.
        Default "San_francisco".
    colors : list or None, optional
        Optional list of color strings/hex codes to use for plotting each distinct
        GCP. If None a default palette is chosen. Must have at least as many
        entries as distinct GCPs (otherwise a ValueError is raised).
    device : torch.device, optional
        PyTorch device used when constructing tensors and models (e.g. CPU or CUDA).
        Default: torch.device("cpu").
    correction_function_parameters_PSM : dict or None, optional
        Parameters passed to the PSM correction function constructor. If None,
        sensible defaults are used: {"linear_constraint": 1e-4,
        "quadratic_constraint": 1e-12, "no_parameters": 7}.
    correction_function_parameters_RFM : dict or None, optional
        Parameters passed to the RFM correction function constructor. If None,
        sensible defaults are used: {"linear_constraint": 1,
        "quadratic_constraint": 1, "no_parameters": 80}.

    Raises
    ------
    AssertionError
        If required files (frame TIFF, PSM/RFM metadata, GCP JSONs, optimization
        JSONs) or required keys inside those files are missing. Also raised if the
        provided `colors` list is too short for the number of GCPs.
    ValueError
        If `colors` is provided but has fewer entries than the number of GCPs.

    Behavior / Side effects
    -----------------------
    - Loads the frame TIFF as a single-band image and scales it for display.
    - Reads PSM metadata from "../../{city}/l1a_frames/{frame_path}_pinhole.json".
    - Reads RFM model metadata from "../../{city}/frameRPC.json".
    - Reads GCP metadata from "../../{city}/own_GCPs/GCPs.json" and image positions
      from "../../{city}/own_GCPs/image_position.json".
    - If `show_optimized_GCPs` is True, loads optimized parameters from files in
      "../../optimization/" named "{optimized_function}_PSM.json" and
      "{optimized_function}_RFM.json" and reconstructs corrected models.
    - Plots:
        - Projected PSM positions (marker "x") and RFM positions (marker "+").
        - Optimized/corrected PSM positions (marker "o") and RFM positions (marker "*").
        - Real annotated GCP positions (marker "v" if used in training, "^" if control).
    - Shows a legend and blocks until the Matplotlib window is closed.

    Returns
    -------
    None
        The function produces an interactive Matplotlib figure and returns None.

    Example
    -------
    show_GCPs_on_frame(
        "1293562080.02321601_sc00113_c1_PAN_i0000000185",
        optimized_function="shift",
        method="gradient",
        city="San_francisco",
        train_GCPs={"San_francisco": ["1"]},
        model="both"
    )
    """
    assert model in ["PSM", "RFM", "both"], "Model must be 'PSM', 'RFM', or 'both'."
    assert os.path.exists(f"../../{city}/l1a_frames/"+frame_path+".tif"), f"Frame {frame_path} does not exist in city {city}."

    with rasterio.open(f"../../{city}/l1a_frames/"+frame_path+".tif") as src:
        img = src.read(1).astype(np.float32)

        img_scaled = (img - img.min()) / (img.max() - img.min()) * 255
        img_scaled = img_scaled.astype(np.uint8)

        profile = src.profile
        profile.update(dtype=rasterio.uint8)

        dpi = matplotlib.rcParams['figure.dpi']

        height, width = img.shape
        figsize = width / float(dpi), height / float(dpi)

        plt.figure(figsize=figsize)

        plt.imshow(img_scaled, cmap="gray", origin="upper", vmin=np.percentile(img_scaled,2), vmax=np.percentile(img_scaled,98))

    if (model == "PSM" or model == "both") and correction_function_parameters_PSM is None:
        correction_function_parameters_PSM = {"linear_constraint": 1e-4, "quadratic_constraint": 1e-12, "no_parameters": 7}
    if (model == "RFM" or model == "both") and correction_function_parameters_RFM is None:
        correction_function_parameters_RFM = {"linear_constraint": 1, "quadratic_constraint": 1, "no_parameters": 80}

    assert os.path.exists(f"../../{city}/own_GCPs/GCPs.json"), f"GCPs data does not exist for {city}."
    with open(f"../../{city}/own_GCPs/GCPs.json", "r") as f:
        GCPinfo = json.load(f)

    if model == "PSM" or model == "both":
        assert os.path.exists(f"../../{city}/l1a_frames/"+frame_path+"_pinhole.json"), f"PSM data does not exist for frame {frame_path} in {city}."
        with open(f"../../{city}/l1a_frames/" + frame_path + "_pinhole.json", "r") as f:
            FramePSMinfo = json.load(f)

        assert "P_projective" in FramePSMinfo and "P_camera" in FramePSMinfo and "P_intrinsic" in FramePSMinfo, f"PSM data for frame {frame_path} in {city} is missing required matrices."
    if model == "RFM" or model == "both":
        assert os.path.exists(f"../../{city}/frameRPC.json"), f"RFM data does not exist for city {city}."

        with open(f"../../{city}/frameRPC.json", "r") as f:
            FrameRFMinfo = json.load(f)

        assert frame_path in FrameRFMinfo, f"Frame {frame_path} does not exist in RFM data for city {city}."

        FrameRFMinfo = FrameRFMinfo[frame_path]


    assert os.path.exists(f"../../{city}/own_GCPs/image_position.json"), f"GCPs image position data does not exist for city {city}."

    with open(f"../../{city}/own_GCPs/image_position.json", "r") as f:
        FrameGCPs = json.load(f)

    if frame_path in FrameGCPs:
        GCPs = FrameGCPs[frame_path]["GCPs"].keys()
    else:
        GCPs = []

    if colors is None:

        if len(GCPs) <= 10:
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        else:
            cmap = matplotlib.colormaps.get_cmap('viridis')
            samples = np.linspace(0, 1, len(GCPs), endpoint=True)
            colors = [mcolors.to_hex(cmap(i)) for i in samples]

    elif len(GCPs) > len(colors):
        raise ValueError(f"Not enough colors provided for the number of GCPs. {len(GCPs)} GCPs but only {len(colors)} colors.")

    GCPs_colors = {GCP: colors[i % len(colors)] for i, GCP in enumerate(GCPs)}

    realGCPsposition = {}

    if train_GCPs is None:
        cities = supported_cities
        train_GCPs = {}

        for ct in cities:
            with open(f"../../{ct}/own_GCPs/image_position.json", "r") as f:
                realGCPsposition[ct] = json.load(f)

            train_GCPs[ct] = []
            for frame in realGCPsposition[ct]:
                for GCP in realGCPsposition[ct][frame]["GCPs"]:
                    train_GCPs[ct].append(GCP)

    else:
        cities = train_GCPs.keys()
        for ct in cities:
            with open(f"../../{ct}/own_GCPs/image_position.json", "r") as f:
                realGCPsposition[ct] = json.load(f)

    for ct in cities:
        with open(f"../../{ct}/own_GCPs/image_position.json", "r") as f:
            realGCPsposition[ct] = json.load(f)

    if model == "both" or model == "PSM":
        P_projective = torch.tensor(FramePSMinfo["P_projective"], dtype=torch.float64)

        P_camera = torch.tensor(FramePSMinfo["P_camera"], dtype=torch.float64)
        P_intrinsic = torch.tensor(FramePSMinfo["P_intrinsic"], dtype=torch.float64)

    if show_projected_GCPs:
        if(model == "both" or model == "RFM"):
            RFM_model = RFM(FrameRFMinfo["LAT_OFF"], FrameRFMinfo["LAT_SCALE"], FrameRFMinfo["LONG_OFF"],
                        FrameRFMinfo["LONG_SCALE"], FrameRFMinfo["HEIGHT_OFF"], FrameRFMinfo["HEIGHT_SCALE"],
                        FrameRFMinfo["LINE_OFF"], FrameRFMinfo["LINE_SCALE"], FrameRFMinfo["SAMP_OFF"],
                        FrameRFMinfo["SAMP_SCALE"], torch.tensor(FrameRFMinfo["LINE_NUM_COEFFS"], dtype=torch.float64, device=device),
                        torch.tensor(FrameRFMinfo["LINE_DEN_COEFFS"], dtype=torch.float64, device=device),
                        torch.tensor(FrameRFMinfo["SAMP_NUM_COEFFS"], dtype=torch.float64, device=device),
                        torch.tensor(FrameRFMinfo["SAMP_DEN_COEFFS"], dtype=torch.float64, device=device), numpy=False, device=device)
        for GCP in GCPs:
            meta_data = GCPinfo[GCP]
            x_ecef = float(meta_data["x_ecef"])
            y_ecef = float(meta_data["y_ecef"])
            z_ecef = float(meta_data["z_ecef"])

            B = float(meta_data["lat"])
            L = float(meta_data["lon"])
            H = float(meta_data["alt"])

            if model == "both" or model == "PSM":
                im_space = P_projective @ torch.tensor([x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64, device=device)

                im_x = im_space[0] / im_space[2]
                im_y = im_space[1] / im_space[2]

                plt.scatter(im_x, im_y, color=GCPs_colors[GCP], marker="x", label=f"{GCP} PSM", s=100)

            if (model == "both" or model == "RFM"):
                im_x, im_y = RFM_model(B, L, H)
                if model == "both" or model == "RFM":
                    plt.scatter(im_x, im_y, color=GCPs_colors[GCP], marker="+", label=f"{GCP} RFM", s=100)

    if show_optimized_GCPs:
        if model == "both" or model == "RFM":

            assert os.path.exists(f"../../optimization/" + optimized_function.__name__ + "_RFM.json"), f"Optimized RFM parameters for function {optimized_function} do not exist."
            with open(f"../../optimization/" + optimized_function.__name__ + "_RFM.json", "r") as f:
                optimized_results_RFM = json.load(f)

            assert method in optimized_results_RFM, f"Method {method} not found in optimized RFM results."
            assert str(correction_function_parameters_RFM) in optimized_results_RFM[method], f"Correction function parameters {correction_function_parameters_RFM} not found in optimized RFM results for method {method}."
            assert str(train_GCPs) in optimized_results_RFM[method][str(correction_function_parameters_RFM)], f"Train GCPs {train_GCPs} not found in optimized RFM results for method {method}."

            params = optimized_results_RFM[method][str(correction_function_parameters_RFM)][str(train_GCPs)]

            params = torch.tensor(params, dtype=torch.float64)

            line_num_coeffs = torch.tensor(FrameRFMinfo["LINE_NUM_COEFFS"], dtype=torch.float64, device=device)
            line_den_coeffs = torch.tensor(FrameRFMinfo["LINE_DEN_COEFFS"], dtype=torch.float64, device=device)
            samp_num_coeffs = torch.tensor(FrameRFMinfo["SAMP_NUM_COEFFS"], dtype=torch.float64, device=device)
            samp_den_coeffs = torch.tensor(FrameRFMinfo["SAMP_DEN_COEFFS"], dtype=torch.float64, device=device)
            args = torch.cat([line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs])

            correction_function_RFM = optimized_function(args, numpy=False, device=device, **correction_function_parameters_RFM)

            line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs = torch.split(correction_function_RFM(params), [20, 20, 20, 20])

            RFM_model = RFM(FrameRFMinfo["LAT_OFF"], FrameRFMinfo["LAT_SCALE"], FrameRFMinfo["LONG_OFF"],
                    FrameRFMinfo["LONG_SCALE"], FrameRFMinfo["HEIGHT_OFF"], FrameRFMinfo["HEIGHT_SCALE"],
                    FrameRFMinfo["LINE_OFF"], FrameRFMinfo["LINE_SCALE"], FrameRFMinfo["SAMP_OFF"],
                    FrameRFMinfo["SAMP_SCALE"], line_num_coeffs, line_den_coeffs, samp_num_coeffs, samp_den_coeffs, numpy=False, device=device)

        if model == "both" or model == "PSM":

            assert os.path.exists(f"../../optimization/" + optimized_function.__name__ + "_PSM.json"), f"Optimized PSM parameters for function {optimized_function} do not exist."

            with open(f"../../optimization/" + optimized_function.__name__ + "_PSM.json", "r") as f:
                optimized_results_PSM = json.load(f)

            assert method in optimized_results_PSM, f"Method {method} not found in optimized PSM results."
            assert str(correction_function_parameters_PSM) in optimized_results_PSM[method], f"Correction function parameters {correction_function_parameters_PSM} not found in optimized PSM results for method {method}."
            assert str(train_GCPs) in optimized_results_PSM[method][str(correction_function_parameters_PSM)], f"Train GCPs {train_GCPs} not found in optimized PSM results for method {method}."

            corrected_exterior_rotation = optimized_results_PSM[method][str(correction_function_parameters_PSM)][str(train_GCPs)]

            original_exterior_rotation = FramePSMinfo["exterior_orientation"]

            quaternion = torch.tensor(
            [original_exterior_rotation["qw_ecef"], original_exterior_rotation["qx_ecef"],
             original_exterior_rotation["qy_ecef"],
             original_exterior_rotation["qz_ecef"]], dtype=torch.float64, device=device)

            sat_position = torch.tensor(
            [original_exterior_rotation["x_ecef_meters"], original_exterior_rotation["y_ecef_meters"],
             original_exterior_rotation["z_ecef_meters"]], dtype=torch.float64, device=device)

            corrected_exterior_rotation = torch.tensor(corrected_exterior_rotation, dtype=torch.float64, device=device)

            correction_function = optimized_function(torch.cat([quaternion, sat_position]),
                                                            numpy=False, **correction_function_parameters_PSM, device=device)

            corrected_parameters = correction_function(corrected_exterior_rotation)

            quaternion, sat_position = torch.split(corrected_parameters, [4, 3])

            P_extrinsic = create_extrinsic(quaternion, sat_position, numpy=False, device=device)

        for GCP in GCPs:

                meta_data = GCPinfo[GCP]
                x_ecef = float(meta_data["x_ecef"])
                y_ecef = float(meta_data["y_ecef"])
                z_ecef = float(meta_data["z_ecef"])

                B = float(meta_data["lat"])
                L = float(meta_data["lon"])
                H = float(meta_data["alt"])

                if model == "both" or model == "PSM":
                    im_space = P_camera @ P_intrinsic @ P_extrinsic @ torch.tensor([x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64, device=device)

                    im_x = im_space[0] / im_space[2]
                    im_y = im_space[1] / im_space[2]
                    plt.scatter(im_x, im_y, color=GCPs_colors[GCP], marker="o", label=f"{GCP} (corrected position PSM)", s=100)

                if model == "both" or model == "RFM":
                    im_x, im_y = RFM_model(B, L, H)
                    plt.scatter(im_x, im_y, color=GCPs_colors[GCP], marker="*", label=f"{GCP} (corrected position RFM)", s=100)

    if show_real_GCPs:
        for GCP in realGCPsposition[city][frame_path]["GCPs"]:
            real_im_x = realGCPsposition[city][frame_path]["GCPs"][GCP]["col"]
            real_im_y = realGCPsposition[city][frame_path]["GCPs"][GCP]["row"]

            if GCP not in train_GCPs[city]:
                marker = "^"
            else:
                marker = "v"

            plt.scatter(real_im_x, real_im_y, color=GCPs_colors[GCP], marker=marker, label=f"{GCP} (real position) "+("control" if GCP not in train_GCPs[city] else "used for correction"), s=100)

    plt.tight_layout()
    plt.subplots_adjust(right=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20)
    plt.title(f"corrected {optimized_function.__name__} function with {method} method", fontsize=20)
    plt.show()

if __name__ == "__main__":

    # show_GCPs_on_frame("1293376734.34837317_sc00113_c1_PAN_i0000000200", show_real_GCPs=False, show_optimized_GCPs=False, show_projected_GCPs=False, city="Cocabamba")

    # show_GCPs_on_frame("1293562079.26564479_sc00113_c1_PAN_i0000000150", method_PSM="gradient", optimized_function="shift")


    show_GCPs_on_frame("1293562080.02321601_sc00113_c1_PAN_i0000000185", optimized_function=shift, method="gradient", city="San_francisco", train_GCPs={"San_francisco": ["1"]}, model="both")
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
