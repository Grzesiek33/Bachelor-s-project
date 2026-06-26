import os
import warnings

from scipy.optimize import minimize
import json

from src.utils.PSM_model import *
from src.optimize.correction_functions import *
from src.optimize.optimization_functions import MSE
import torch
import numpy as np
from src.utils.RFM_model import *
from src.utils.cities import supported_cities
from itertools import combinations

def automated_optimize(correction_model = None, correction_for = "c1", model = "PSM", train_set = None, supress_warnings=True, device=torch.device("cpu")):
    """Automatically optimize camera parameters across multiple training set combinations.

        This function orchestrates the optimization process by generating combinations of ground control points (GCPs)
        and calling optimize_camera_parameters for each combination. It automates the process of testing different
        subsets of GCPs to find optimal correction model parameters.

        Parameters
        ----------
        correction_model : dict, optional
            Dictionary describing the correction model and optimization settings (same structure as in optimize_camera_parameters).
            If None, default parameters are used. Expected keys include: "correction_function", "initial_params",
            "optimization_function", "method", "epochs", "lr", "q_constraint", and "correction_function_parameters".
            Default: None

        correction_for : str, optional
            Filter for frame selection. Controls which frames' GCPs are included in training:
            - "all": includes all frames from all cities
            - "c1", "c2", "c3": includes only frames where the camera identifier matches (e.g., "c1" for camera 1)
            - frame identifiers - list of strings: includes only frames whose names match the provided list of frame identifiers
            Default: "c1"

        model : {"PSM", "RFM"}, optional
            The camera model to optimize.
            - "PSM": Physical Sensor Model (optimizes quaternion and satellite position)
            - "RFM": Rational Function Model (optimizes RFM polynomial coefficients)
            Default: "PSM"

        train_set : dict or None, optional
            Dictionary mapping city names to lists of combination sizes to generate.
            Example: {"San_francisco": [1, 2], "Cocabamba": [1]} will test all combinations of size 1 and 2 for San Francisco,
            and all combinations of size 1 for Cocabamba.
            If None (default), all combinations of single GCPs for each city are generated and tested.

        supress_warnings : bool, optional
            If True, suppresses warnings when default values are used for missing correction_model entries.
            If False, warnings are issued for missing parameters.
            Default: True

        device : torch.device, optional
            PyTorch device for computations (e.g., torch.device("cpu") or torch.device("cuda")).
            Used when gradient-based optimization is selected.
            Default: torch.device("cpu")

        Behavior
        --------
        1. Loads GCP position data from all supported cities' own_GCPs/image_position.json files
        2. Filters GCPs based on the correction_for parameter (all frames, specific camera, or specific frame)
        3. Generates the train_set dictionary if not provided (all combinations of size 1 for each city)
        4. For each city and each combination size in train_set:
           - Generates all possible combinations of GCPs of that size
           - Calls optimize_camera_parameters for each combination
           - Results are saved to ../../optimization/{correction_function_name}_{model}.json

        Returns
        -------
        None
            Results are saved to the optimization JSON file. A completion message is printed by optimize_camera_parameters
            for each combination processed.

        Raises
        ------
        ValueError
            If an unsupported model is provided (raised by optimize_camera_parameters).
        FileNotFoundError
            If required GCP position JSON files cannot be found.

        Examples
        --------
        Optimize PSM using shift correction, testing all combinations of size 1 for San Francisco:
        automated_optimize(
             correction_model={"correction_function": shift},
             model="PSM",
             train_set={"San_francisco": [1]},
             device=torch.device("cpu")
             )

        Optimize RFM using linear correction for all camera 1 frames, testing combinations of sizes 1-3:
         automated_optimize(
             correction_model={"correction_function": linear},
             correction_for="c1",
             model="RFM",
             train_set={"San_francisco": [1, 2, 3], "Cocabamba": [1, 2, 3]},
             device=torch.device("cuda")
             )

        Optimize PSM for a specific frame, testing all combination sizes:
         automated_optimize(
             correction_model={"correction_function": shift},
             correction_for=["1293562080.02321601_sc00113_c1_PAN_i0000000185"],
             model="PSM",
             train_set=None  # All combinations will be generated
             )

        """
    realGCPsposition = {}
    train_GCPs = {}

    cities = supported_cities
    for city in cities:
        with open(f"../../{city}/own_GCPs/image_position.json", "r") as f:
            realGCPsposition[city] = json.load(f)

        train_GCPs[city] = []
        for frame in realGCPsposition[city]:
            if correction_for == "all" or (correction_for[0] == "c" and frame.split("_")[2] == correction_for) or (
                    frame in correction_for):
                for GCP in realGCPsposition[city][frame]["GCPs"]:
                    train_GCPs[city].append(GCP)



    if train_set is None:
        train_set = {city: [1] for city in supported_cities}

    for city in train_set:
        for i in train_set[city]:
            for comb in combinations(train_GCPs[city], i):
                optimize_camera_parameters({city: list(comb)}, correction_model, model, supress_warnings, device=device)

def optimize_camera_parameters(train_GCPs = "all", correction_model = None, model = "PSM", supress_warnings=True, device=torch.device("cpu")):
    """Optimize camera parameters using ground control points (GCPs) and a correction function.
    This function fits a correction model that adjusts model coefficients so that projected image coordinates of known ground control points match their measured image coordinates. It supports two underlying models:
    "PSM" (physical sensor model): optimizes a quaternion (camera rotation) and satellite position using the provided correction function. A soft constraint on the quaternion norm may be added to the loss (via q_constraint).
    "RFM" (rational function model): optimizes the RFM polynomial coefficients (four 20-coefficient vectors by default) using the provided correction function.

    Parameters
    train_GCPs : "all" or dict, optional
    If "all" (default), all GCPs from all supported cities (as listed in src.utils.cities.supported_cities) are used for training. If a dict is provided, it must map city names to lists of GCP identifiers (strings) to use for training. Example: {"San_francisco": ["1", "7"], "Cocabamba": ["3"]}.
    correction_model : dict, optional
    Dictionary describing the correction model and optimization settings. Expected keys: - "correction_function": a callable factory that, given current frame parameters (quaternion + sat position for PSM, or RFM coeffs and RPC for RFM), returns a function that maps optimization parameters -> corrected parameters. Both a PyTorch (differentiable) and a NumPy variant are supported; which is used depends on the chosen optimization method. - "initial_params": initial parameter vector for optimization (NumPy array or list). - "optimization_function": callable loss function taking (col, row, im_x, im_y) and returning a scalar error (default: MSE). - "correction_function_parameters": dict of extra kwargs passed to the correction function factory (e.g. constraints, number of parameters). - "method": optimization method. If "gradient", PyTorch's optimizer (Adam) is used and the code expects differentiable (torch) correction functions. Otherwise any supported scipy method (e.g. "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP") is allowed and NumPy-based correction functions are used. - "epochs": number of gradient steps (or max iterations for scipy). - "lr": learning rate for gradient optimization. - "q_constraint": (PSM only) float weight for the quaternion unit-length soft constraint.
    model : {"PSM", "RFM"}, optional
    Which model to optimize. Default is "PSM".
    supress_warnings : bool, optional
    If False, the function will warn when default values are used for missing correction_model entries. Default True.
    device : torch.device, optional Device for PyTorch computations when using gradient-based optimization (e.g. torch.device("cpu") or CUDA device).

    Behavior
    Loads GCP data, measured image positions, and per-frame parameters from each city's own_GCPs and frameRPC.json and frame pinhole JSON files.
    Constructs an objective function that:
    for each training GCP in each frame, projects the ground coordinates using the current model parameters (PSM or RFM) after they are adjusted by the correction function,
    computes the error with optimization_function and accumulates it.
    for PSM, a quaternion norm penalty is added if q_constraint is present.
    Writes optimization results to a JSON file at: ../../optimization/{correction_function.__name__}_{model}.json. The JSON structure is keyed by optimization method, stringified correction_function_parameters and stringified train_GCPs, with the stored value being the optimized parameter vector.
    Returns
    None
    Results are saved to the optimization JSON file and a completion message is printed.
    Raises
    ValueError If an unsupported model or unsupported/non-listed method is provided.

    Examples
    Use gradient optimization to fit a small shift correction for PSM:

    opt = {"correction_function": shift, "initial_params": shift_initial_params_PSM, "method": "gradient", "epochs": 1000, "lr": 1e-9}
    optimize_camera_parameters(train_GCPs="all", correction_model=opt, model="PSM", device=torch.device("cpu"))

    Use scipy Nelder-Mead to fit linear correction for RFM on a specific city subset:
    opt = {"correction_function": linear, "initial_params": linear_initial_params_RFM, "method": "Nelder-Mead", "epochs": 2000}
    optimize_camera_parameters(train_GCPs={"San_francisco": ["1","2"]}, correction_model=opt, model="RFM")

    """

    if correction_model is None:
        if model == "PSM":
            correction_model = {"correction_function": linear, "initial_params": zero_based_initial_params("linear", 7), "optimization_function": MSE, "q_constraint": 1, "correction_function_parameters": {"linear_constraint": 1e-4, "quadratic_constraint": 1e-12, "no_parameters": 7}, "method": "gradient", "lr":1e-9, "epochs":1000}
        elif model == "RFM":
            correction_model = {"correction_function": linear, "initial_params": zero_based_initial_params("linear", 80), "optimization_function": MSE, "q_constraint": 1, "correction_function_parameters": {"linear_constraint": 1, "quadratic_constraint": 1, "no_parameters": 80}, "method": "gradient", "lr":1e-5, "epochs":1000}
        else:
            raise ValueError(f"model {model} not implemented, did you mean 'PSM' or 'RFM'?")
    else:
        if "method" not in correction_model:
            if not supress_warnings:
                warnings.warn(
                    "Method not specified, using gradient instead",
                    UserWarning
                )
            correction_model["method"] = "gradient"
        elif correction_model["method"] != "gradient" and correction_model["method"] not in ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]:
            raise ValueError(f"Method {correction_model['method']} not implemented, did you mean 'gradient', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA' or 'SLSQP'?")

        if "epochs" not in correction_model:
            if not supress_warnings:
                warnings.warn("no. of epochs not specified, using 1000 instead. Note that for non-gradient methods, this denotes the maximum number of iterations", UserWarning)
            correction_model["epochs"] = 1000

        if "correction_function" not in correction_model:
            if not supress_warnings:
                warnings.warn("Correction function not specified, using linear instead", UserWarning)
            correction_model["correction_function"] = linear
        if "initial_params" not in correction_model:
            if not supress_warnings:
                warnings.warn("Initial parameters not specified, using zero based initial parameters", UserWarning)
            if model == "PSM":
                correction_model["initial_params"] = zero_based_initial_params(correction_model["correction_function"].__name__, 7)
            elif model == "RFM":
                correction_model["initial_params"] = zero_based_initial_params(correction_model["correction_function"].__name__, 80)

        if "optimization_function" not in correction_model:
            if not supress_warnings:
                warnings.warn("Optimization function not specified, using MSE instead", UserWarning)
            correction_model["optimization_function"] = MSE

        if "correction_function_parameters" not in correction_model:
            if not supress_warnings:
                warnings.warn("Correction function parameters not specified, using default parameters", UserWarning)
            if model == "PSM":
                correction_model["correction_function_parameters"] = {"linear_constraint": 1e-4, "quadratic_constraint": 1e-12, "no_parameters": 7}
            elif model == "RFM":
                correction_model["correction_function_parameters"] = {"linear_constraint": 1, "quadratic_constraint": 1, "no_parameters": 80}

        if model == "PSM" and "q_constraint" not in correction_model:
            if not supress_warnings:
                warnings.warn("q_constraint not specified, using 1 instead", UserWarning)
            correction_model["q_constraint"] = 1

    GCPinfo = {}
    realGCPsposition = {}
    frameInfo = {}

    if train_GCPs == "all":
        train_GCPs = {}
        cities = supported_cities
        for city in cities:
            with open(f"../../{city}/own_GCPs/GCPs.json", "r") as f:
                GCPinfo[city] = json.load(f)

            with open(f"../../{city}/own_GCPs/image_position.json", "r") as f:
                realGCPsposition[city] = json.load(f)

            with open(f"../../{city}/frameRPC.json", "r") as f:
                frameInfo[city] = json.load(f)

            train_GCPs[city] = []
            for frame in realGCPsposition[city]:
                for GCP in realGCPsposition[city][frame]["GCPs"]:
                    train_GCPs[city].append(GCP)
    else:
        cities = train_GCPs.keys()
        for city in cities:
            with open(f"../../{city}/own_GCPs/GCPs.json", "r") as f:
                GCPinfo[city] = json.load(f)

            with open(f"../../{city}/own_GCPs/image_position.json", "r") as f:
                realGCPsposition[city] = json.load(f)

            with open(f"../../{city}/frameRPC.json", "r") as f:
                frameInfo[city] = json.load(f)

    if model == "PSM":
        if correction_model["method"] == "gradient":

            def objective_function(params):
                total_error = torch.tensor(0, dtype=torch.float64, device=device)
                for ct in cities:
                    for fr in realGCPsposition[ct]:
                        with open(f"../../{ct}/l1a_frames/" + fr + "_pinhole.json", "r") as f:
                            Frameinfo = json.load(f)

                        P_camera = torch.tensor(Frameinfo["P_camera"], dtype=torch.float64, device=device)
                        P_intrinsic = torch.tensor(Frameinfo["P_intrinsic"], dtype=torch.float64, device=device)

                        exterior_rotation = Frameinfo["exterior_orientation"]

                        quaternion = torch.tensor(
                            [exterior_rotation["qw_ecef"], exterior_rotation["qx_ecef"],
                             exterior_rotation["qy_ecef"],
                             exterior_rotation["qz_ecef"]], dtype=torch.float64, device=device)

                        sat_position = torch.tensor(
                            [exterior_rotation["x_ecef_meters"], exterior_rotation["y_ecef_meters"],
                             exterior_rotation["z_ecef_meters"]], dtype=torch.float64, device=device)

                        correction_function = correction_model["correction_function"](torch.cat([quaternion, sat_position]),
                                                     numpy=False, device=device, **correction_model["correction_function_parameters"])

                        corrected_parameters = correction_function(params)

                        q, sat_pos = torch.split(corrected_parameters, [4, 3])

                        model = PSM(P_camera, P_intrinsic, q, sat_pos, numpy=False, device=device)

                        for GCP in realGCPsposition[ct][fr]["GCPs"]:
                            if GCP in train_GCPs[ct]:
                                x_ecef = float(GCPinfo[ct][GCP]["x_ecef"])
                                y_ecef = float(GCPinfo[ct][GCP]["y_ecef"])
                                z_ecef = float(GCPinfo[ct][GCP]["z_ecef"])

                                im_x, im_y = model(x_ecef, y_ecef, z_ecef)

                                row = realGCPsposition[ct][fr]["GCPs"][GCP]["row"]
                                col = realGCPsposition[ct][fr]["GCPs"][GCP]["col"]

                                total_error = total_error + correction_model["optimization_function"](col, row, im_x, im_y) + correction_model["q_constraint"] * ((1 - ( q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 )) ** 2)

                return total_error

        else:
            def objective_function(params):
                total_error = 0
                for ct in cities:
                    for fr in realGCPsposition[ct]:
                        with open(f"../../{ct}/l1a_frames/" + fr + "_pinhole.json", "r") as f:
                            Frameinfo = json.load(f)

                        P_camera = np.array(Frameinfo["P_camera"], dtype=np.float64)
                        P_intrinsic = np.array(Frameinfo["P_intrinsic"], dtype=np.float64)

                        exterior_rotation = Frameinfo["exterior_orientation"]

                        quaternion = np.array(
                            [exterior_rotation["qw_ecef"], exterior_rotation["qx_ecef"],
                             exterior_rotation["qy_ecef"],
                             exterior_rotation["qz_ecef"]], dtype=np.float64)

                        sat_position = np.array(
                            [exterior_rotation["x_ecef_meters"], exterior_rotation["y_ecef_meters"],
                             exterior_rotation["z_ecef_meters"]], dtype=np.float64)

                        correction_function = correction_model["correction_function"](np.concat([quaternion, sat_position]), numpy=True, **correction_model["correction_function_parameters"])

                        corrected_parameters = correction_function(params)

                        q, sat_pos = corrected_parameters[:4], corrected_parameters[4:]

                        model = PSM(P_camera, P_intrinsic, q, sat_pos, numpy=True)

                        for GCP in realGCPsposition[ct][fr]["GCPs"]:
                            if GCP in train_GCPs[ct]:
                                x_ecef = float(GCPinfo[ct][GCP]["x_ecef"])
                                y_ecef = float(GCPinfo[ct][GCP]["y_ecef"])
                                z_ecef = float(GCPinfo[ct][GCP]["z_ecef"])

                                im_x, im_y = model(x_ecef, y_ecef, z_ecef)

                                row = realGCPsposition[ct][fr]["GCPs"][GCP]["row"]
                                col = realGCPsposition[ct][fr]["GCPs"][GCP]["col"]

                                total_error = total_error + correction_model["optimization_function"](col, row, im_x, im_y) + correction_model["q_constraint"] * ((1 - (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)) ** 2)

                return total_error

    elif model == "RFM":

        if correction_model["method"] == "gradient":

            def objective_function(params):
                total_error = 0
                for ct in cities:
                    for fr in realGCPsposition[ct]:
                        Line_num_coeffs = torch.tensor(frameInfo[ct][fr][f"LINE_NUM_COEFFS"], dtype=torch.float64, device=device)
                        Line_den_coeffs = torch.tensor(frameInfo[ct][fr][f"LINE_DEN_COEFFS"], dtype=torch.float64, device=device)
                        Samp_num_coeffs = torch.tensor(frameInfo[ct][fr][f"SAMP_NUM_COEFFS"], dtype=torch.float64, device=device)
                        Samp_den_coeffs = torch.tensor(frameInfo[ct][fr][f"SAMP_DEN_COEFFS"], dtype=torch.float64, device=device)

                        line_off = frameInfo[ct][fr]["LINE_OFF"]
                        samp_off = frameInfo[ct][fr]["SAMP_OFF"]
                        lat_off = frameInfo[ct][fr]["LAT_OFF"]
                        long_off = frameInfo[ct][fr]["LONG_OFF"]
                        height_off = frameInfo[ct][fr]["HEIGHT_OFF"]
                        line_scale = frameInfo[ct][fr]["LINE_SCALE"]
                        samp_scale = frameInfo[ct][fr]["SAMP_SCALE"]
                        lat_scale = frameInfo[ct][fr]["LAT_SCALE"]
                        long_scale = frameInfo[ct][fr]["LONG_SCALE"]
                        height_scale = frameInfo[ct][fr]["HEIGHT_SCALE"]

                        correction_function = correction_model["correction_function"](
                            torch.cat([Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs]),
                            numpy=False, device=device, **correction_model["correction_function_parameters"])

                        corrected_parameters = correction_function(params)

                        Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs = torch.split(corrected_parameters, [20, 20, 20, 20])

                        model = RFM(lat_off, lat_scale, long_off, long_scale, height_off, height_scale,
                                          line_off, line_scale, samp_off, samp_scale, Line_num_coeffs,
                                          Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs, numpy=False, device=device)

                        for GCP in realGCPsposition[ct][fr]["GCPs"]:
                            if GCP in train_GCPs[ct]:
                                B = float(GCPinfo[ct][GCP]["lat"])
                                L = float(GCPinfo[ct][GCP]["lon"])
                                H = float(GCPinfo[ct][GCP]["alt"])

                                row = realGCPsposition[ct][fr]["GCPs"][GCP]["row"]
                                col = realGCPsposition[ct][fr]["GCPs"][GCP]["col"]

                                im_x, im_y = model(B, L, H)

                                total_error = total_error + correction_model["optimization_function"](col, row, im_x, im_y)

                return total_error

        else:
            def objective_function(params):
                total_error = 0
                for ct in cities:
                    for fr in realGCPsposition[ct]:

                        Line_num_coeffs = np.array(frameInfo[ct][fr][f"LINE_NUM_COEFFS"], dtype=np.float64)
                        Line_den_coeffs = np.array(frameInfo[ct][fr][f"LINE_DEN_COEFFS"], dtype=np.float64)
                        Samp_num_coeffs = np.array(frameInfo[ct][fr][f"SAMP_NUM_COEFFS"], dtype=np.float64)
                        Samp_den_coeffs = np.array(frameInfo[ct][fr][f"SAMP_DEN_COEFFS"], dtype=np.float64)

                        line_off = frameInfo[ct][fr]["LINE_OFF"]
                        samp_off = frameInfo[ct][fr]["SAMP_OFF"]
                        lat_off = frameInfo[ct][fr]["LAT_OFF"]
                        long_off = frameInfo[ct][fr]["LONG_OFF"]
                        height_off = frameInfo[ct][fr]["HEIGHT_OFF"]
                        line_scale = frameInfo[ct][fr]["LINE_SCALE"]
                        samp_scale = frameInfo[ct][fr]["SAMP_SCALE"]
                        lat_scale = frameInfo[ct][fr]["LAT_SCALE"]
                        long_scale = frameInfo[ct][fr]["LONG_SCALE"]
                        height_scale = frameInfo[ct][fr]["HEIGHT_SCALE"]

                        correction_function = correction_model["correction_function"](
                            np.concat([Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs]),
                            numpy=True, **correction_model["correction_function_parameters"])

                        corrected_parameters = correction_function(params)

                        Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs = corrected_parameters.reshape(4, 20)

                        model = RFM(lat_off, lat_scale, long_off, long_scale, height_off, height_scale,
                                          line_off, line_scale, samp_off, samp_scale, Line_num_coeffs,
                                          Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs, numpy=True, device=device)

                        for GCP in realGCPsposition[ct][fr]["GCPs"]:
                            if GCP in train_GCPs[ct]:
                                B = float(GCPinfo[ct][GCP]["lat"])
                                L = float(GCPinfo[ct][GCP]["lon"])
                                H = float(GCPinfo[ct][GCP]["alt"])

                                row = realGCPsposition[ct][fr]["GCPs"][GCP]["row"]
                                col = realGCPsposition[ct][fr]["GCPs"][GCP]["col"]

                                im_x, im_y = model(B, L, H)

                                total_error = total_error + correction_model["optimization_function"](col, row, im_x, im_y)

                return total_error
    else:
        raise ValueError(f"model {model} not implemented")

    if correction_model["method"] != "gradient":

        result = minimize(objective_function, correction_model["initial_params"], method=correction_model["method"], options={"maxiter": correction_model["epochs"]})

        optimized_params = result.x.tolist()

    else:
        params = torch.from_numpy(correction_model["initial_params"]).clone().detach().to(device).requires_grad_(True)

        if "lr" not in correction_model:
            if not supress_warnings:
                warnings.warn("Learning rate not specified, using 1e-5 instead", UserWarning)
            correction_model["lr"] = 1e-5

        optimizer = torch.optim.Adam([params], lr=correction_model["lr"])

        for _ in range(correction_model["epochs"]):
            optimizer.zero_grad()
            loss = objective_function(params)
            loss.backward()
            optimizer.step()

        optimized_params = [p.tolist() for p in params]

    if not os.path.exists("../../optimization/"):
        os.makedirs("../../optimization/")

    if not os.path.exists("../../optimization/" + correction_model["correction_function"].__name__ + '_' + model + ".json"):
        with open("../../optimization/" + correction_model["correction_function"].__name__ + '_' + model +  ".json", "w") as f:
            f.write("{}")

    with open("../../optimization/" + correction_model["correction_function"].__name__ + '_' + model + ".json", "r") as f:
        optimized_results = json.load(f)

    if correction_model["method"] not in optimized_results:
        optimized_results[correction_model["method"]] = {}

    if str(correction_model["correction_function_parameters"]) not in optimized_results[correction_model["method"]]:
        optimized_results[correction_model["method"]][str(correction_model["correction_function_parameters"])] = {}

    optimized_results[correction_model["method"]][str(correction_model["correction_function_parameters"])][str(train_GCPs)] = optimized_params

    with open("../../optimization/" + correction_model["correction_function"].__name__ + '_' + model + ".json", "w") as f:
     json.dump(optimized_results, f, indent=2)

    print("Optimization completed")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # automated_optimize(model = "RFM", correction_model={"correction_function": shift}, train_set={"San_francisco": [1,2]}, correction_for=["1293562080.02321601_sc00113_c1_PAN_i0000000185"])
    automated_optimize(model = "PSM", correction_model={"correction_function": shift}, train_set={"San_francisco": [1]}, device=device)
    # automated_optimize(model = "RFM", correction_model={"correction_function": shift}, train_set={"San_francisco": [1]}, device=device)

    # optimize_camera_parameters(model = "RFM", correction_model={"correction_function": shift})
    # optimize_camera_parameters(model = "PSM", correction_model={"correction_function": shift})
    # optimize_camera_parameters(epochs=100, model = "RFM", correction_model=(shift, shift_initial_params_RFM))
    # optimize_camera_parameters(model = "PSM", correction_model=(shift, shift_initial_params_PSM))

    # optimize_camera_parameters(method="gradient", lr=5e-5, epochs=1000, model = "RFM", correction_model=(linear_RFM, linear_initial_params_RFM))
    # optimize_camera_parameters(method="gradient", lr=1e-11, epochs=1000, model = "PSM", correction_model=(linear_PSM, linear_initial_params_PSM))
    # optimize_camera_parameters(model = "RFM", correction_model=(linear_RFM_numpy, linear_initial_params_RFM))
    # optimize_camera_parameters(model = "PSM", correction_model=(linear_PSM_numpy, linear_initial_params_PSM))

    # optimize_camera_parameters(method="gradient", lr=5e-5, model = "RFM", correction_model=(quadratic_RFM_torch, quadratic_initial_params_RFM))
    # optimize_camera_parameters(method="gradient", lr=1e-12, model = "PSM", correction_model=(quadratic_PSM_torch, quadratic_initial_params_PSM))
    # optimize_camera_parameters(model = "RFM", correction_model=(quadratic_RFM_numpy, quadratic_initial_params_RFM))
    # optimize_camera_parameters(model = "PSM", correction_model=(quadratic_PSM_numpy, quadratic_initial_params_PSM))

