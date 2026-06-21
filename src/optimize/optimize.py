import os
import warnings

from scipy.optimize import minimize
import json

from src.utils.PSM_model import *
from correction_functions import *
from optimization_function import MSE
import re
import torch
import numpy as np
from src.utils.RFM_model import *

def optimize_camera_parameters(correction_model = None, correction_for: str = "c1", model = "PSM", cities=None, supress_warnings=True):
    if correction_model is None:
        if model == "PSM":
            correction_model = {"correction_function": linear, "initial_params": zero_based_initial_params("linear", 7), "optimization_function": MSE, "q_constraint": 1, "correction_function_parameters": {"linear_constraint": 1e-4, "quadratic_constraint": 1e-12, "no_parameters": 7}, "method":"gradient", "lr":1e-9, "epochs":1000}
        elif model == "RFM":
            correction_model = {"correction_function": linear, "initial_params": zero_based_initial_params("linear", 80), "optimization_function": MSE, "q_constraint": 1, "correction_function_parameters": {"linear_constraint": 1, "quadratic_constraint": 1, "no_parameters": 80}, "linear_constraint": 1, "quadratic_constraint": 1, "method":"gradient", "lr":1e-5, "epochs":1000}
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

    if cities is None:
        cities = ["San_francisco", "Angkor_wat", "Cochabamba"]

    GCPinfo = {}
    realGCPsposition = {}
    frameInfo = {}

    for city in cities:
        with open(f"../../{city}/own_GCPs/GCPs.json", "r") as f:
            GCPinfo[city] = json.load(f)

        with open(f"../../{city}/own_GCPs/image_position.json", "r") as f:
            realGCPsposition[city] = json.load(f)

        with open(f"../../{city}/frameRPC.json", "r") as f:
            frameInfo[city] = json.load(f)

    control_GCPs = {}

    for city in cities:
        control_GCPs[city] = ""
        for cam in realGCPsposition[city]:
            for frame in realGCPsposition[city][cam]:
                for GCP in realGCPsposition[city][cam][frame]["GCPs"]:
                    if realGCPsposition[city][cam][frame]["GCPs"][GCP]["control"] == 1:
                        control_GCPs[city] += GCP + ", "

    if correction_for[0] == "c":
        if model == "PSM":
            if correction_model["method"] == "gradient":

                def objective_function(params):
                    total_error = torch.tensor(0, dtype=torch.float64)
                    for ct in cities:
                        for fr in realGCPsposition[ct][correction_for]:
                            with open(f"../../{ct}/l1a_frames/" + fr + "_pinhole.json", "r") as f:
                                Frameinfo = json.load(f)

                            P_camera = torch.tensor(Frameinfo["P_camera"], dtype=torch.float64)
                            P_intrinsic = torch.tensor(Frameinfo["P_intrinsic"], dtype=torch.float64)

                            exterior_rotation = Frameinfo["exterior_orientation"]

                            quaternion = torch.tensor(
                                [exterior_rotation["qw_ecef"], exterior_rotation["qx_ecef"],
                                 exterior_rotation["qy_ecef"],
                                 exterior_rotation["qz_ecef"]], dtype=torch.float64)

                            sat_position = torch.tensor(
                                [exterior_rotation["x_ecef_meters"], exterior_rotation["y_ecef_meters"],
                                 exterior_rotation["z_ecef_meters"]], dtype=torch.float64)

                            correction_function = correction_model["correction_function"](torch.cat([quaternion, sat_position]),
                                                         numpy=False, **correction_model["correction_function_parameters"])

                            corrected_parameters = correction_function(params)

                            q, sat_pos = torch.split(corrected_parameters, [4, 3])

                            model = PSM(P_camera, P_intrinsic, q, sat_pos, numpy=False)

                            for GCP in realGCPsposition[ct][correction_for][fr]["GCPs"]:
                                x_ecef = float(GCPinfo[ct][GCP]["x_ecef"])
                                y_ecef = float(GCPinfo[ct][GCP]["y_ecef"])
                                z_ecef = float(GCPinfo[ct][GCP]["z_ecef"])

                                im_x, im_y = model(x_ecef, y_ecef, z_ecef)

                                row = realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["row"]
                                col = realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["col"]

                                total_error = total_error + correction_model["optimization_function"](col, row, im_x, im_y) + correction_model["q_constraint"] * ((1 - ( q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 )) ** 2)

                    return total_error

            else:
                def objective_function(params):
                    total_error = 0
                    for ct in cities:
                        for fr in realGCPsposition[ct][correction_for]:
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

                            for GCP in realGCPsposition[ct][correction_for][fr]["GCPs"]:
                                if realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["control"] == 0:

                                    x_ecef = float(GCPinfo[ct][GCP]["x_ecef"])
                                    y_ecef = float(GCPinfo[ct][GCP]["y_ecef"])
                                    z_ecef = float(GCPinfo[ct][GCP]["z_ecef"])

                                    im_x, im_y = model(x_ecef, y_ecef, z_ecef)

                                    row = realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["row"]
                                    col = realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["col"]

                                    total_error = total_error + correction_model["optimization_function"](col, row, im_x, im_y) + correction_model["q_constraint"] * ((1 - (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)) ** 2)

                    return total_error

        elif model == "RFM":

            if correction_model["method"] == "gradient":

                def objective_function(params):
                    total_error = 0
                    for ct in cities:
                        for fr in realGCPsposition[ct][correction_for]:
                            Line_num_coeffs = torch.tensor(frameInfo[ct][fr][f"LINE_NUM_COEFFS"], dtype=torch.float64)
                            Line_den_coeffs = torch.tensor(frameInfo[ct][fr][f"LINE_DEN_COEFFS"], dtype=torch.float64)
                            Samp_num_coeffs = torch.tensor(frameInfo[ct][fr][f"SAMP_NUM_COEFFS"], dtype=torch.float64)
                            Samp_den_coeffs = torch.tensor(frameInfo[ct][fr][f"SAMP_DEN_COEFFS"], dtype=torch.float64)

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
                                numpy=False, **correction_model["correction_function_parameters"])

                            corrected_parameters = correction_function(params)

                            Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs = torch.split(corrected_parameters, [20, 20, 20, 20])

                            model = RFM_torch(lat_off, lat_scale, long_off, long_scale, height_off, height_scale,
                                              line_off, line_scale, samp_off, samp_scale, Line_num_coeffs,
                                              Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs)

                            for GCP in realGCPsposition[ct][correction_for][fr]["GCPs"]:
                                if realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["control"] == 0:
                                    B = float(GCPinfo[ct][GCP]["lat"])
                                    L = float(GCPinfo[ct][GCP]["lon"])
                                    H = float(GCPinfo[ct][GCP]["alt"])

                                    row = realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["row"]
                                    col = realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["col"]

                                    im_y, im_x = model(B, L, H)

                                    total_error = total_error + correction_model["optimization_function"](col, row, im_x, im_y)

                    return total_error

            else:
                def objective_function(params):
                    total_error = 0
                    for ct in cities:
                        for fr in realGCPsposition[ct][correction_for]:

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

                            model = RFM_numpy(lat_off, lat_scale, long_off, long_scale, height_off, height_scale,
                                              line_off, line_scale, samp_off, samp_scale, Line_num_coeffs,
                                              Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs)

                            for GCP in realGCPsposition[ct][correction_for][fr]["GCPs"]:
                                if realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["control"] == 0:
                                    B = float(GCPinfo[ct][GCP]["lat"])
                                    L = float(GCPinfo[ct][GCP]["lon"])
                                    H = float(GCPinfo[ct][GCP]["alt"])

                                    row = realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["row"]
                                    col = realGCPsposition[ct][correction_for][fr]["GCPs"][GCP]["col"]

                                    im_y, im_x = model(B, L, H)

                                    total_error = total_error + correction_model["optimization_function"](col, row, im_x, im_y)

                    return total_error
        else:
            raise ValueError(f"model {model} not implemented")

    ### TODO: implement for single frame
    # elif "." in correction_for:
    #     if model == "PSM":
    #         def objective_function(params):
    #             total_error = torch.tensor(0, dtype=torch.float64)
    #             with open(f"../../{city}/l1a_frames/" + correction_for + "_pinhole.json", "r") as f:
    #                 Frameinfo = json.load(f)
    #
    #             cam = re.search(r"c[1,2,3]", correction_for).group(0)
    #
    #             for GCP in realGCPsposition[city][cam][correction_for]["GCPs"]:
    #                 if realGCPsposition[city][cam][correction_for]["GCPs"][GCP]["control"] == 0 and (restricted_to is None or GCP in restricted_to) and (exclude is None or GCP not in exclude):
    #                     total_error = total_error + helper_function_PSM(params, realGCPsposition[city][cam][correction_for]["GCPs"][GCP], GCPinfo[city][GCP], Frameinfo, correction_function=correction_model[0], optimization_function=optimization_function, q_constraint=q_constraint, unflatten=correction_model[3], nump=(method!="gradient"))
    #             return total_error
    #     elif model == "RFM":
    #         def objective_function(params):
    #             total_error = torch.tensor(0, dtype=torch.float64)
    #             with open(f"../../{city}/frameRPC.json", "r") as f:
    #                 Frameinfo = json.load(f)[correction_for]
    #
    #             cam = re.search(r"c[1,2,3]", correction_for).group(0)
    #
    #             for GCP in realGCPsposition[city][cam][correction_for]["GCPs"]:
    #                 if realGCPsposition[city][cam][correction_for]["GCPs"][GCP]["control"] == 0 and (restricted_to is None or GCP in restricted_to) and (exclude is None or GCP not in exclude):
    #                     total_error = total_error + helper_function_RFM(params, realGCPsposition[city][cam][correction_for]["GCPs"][GCP], GCPinfo[city][GCP], Frameinfo, correction_function=correction_model[0], optimization_function=optimization_function, unflatten=correction_model[3], nump=(method!="gradient"))
    #             return total_error

    else:
        raise ValueError(f"correction for {correction_for} not implemented")

    if correction_model["method"] != "gradient":

        result = minimize(objective_function, correction_model["initial_params"], method=correction_model["method"], options={"maxiter": correction_model["epochs"]})

        optimized_params = result.x.tolist()

    else:
        params = torch.from_numpy(correction_model["initial_params"]).clone().detach().requires_grad_(True)

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

    if not os.path.exists("../../optimization/" + correction_model["correction_function"].__name__ + '_' + model + ".json"):
        with open("../../optimization/" + correction_model["correction_function"].__name__ + '_' + model +  ".json", "w") as f:
            f.write("{}")

    with open("../../optimization/" + correction_model["correction_function"].__name__ + '_' + model + ".json", "r") as f:
        optimized_results = json.load(f)
    if correction_for not in optimized_results:
        optimized_results[correction_for] = {}

    if correction_model["method"] not in optimized_results[correction_for]:
        optimized_results[correction_for][correction_model["method"]] = {}
    if str(cities) not in optimized_results[correction_for][correction_model["method"]]:
        optimized_results[correction_for][correction_model["method"]][str(cities)] = {}
    if str(control_GCPs) not in optimized_results[correction_for][correction_model["method"]][str(cities)]:
        optimized_results[correction_for][correction_model["method"]][str(cities)][str(control_GCPs)] = {}

    optimized_results[correction_for][correction_model["method"]][str(cities)][str(control_GCPs)] = optimized_params

    with open("../../optimization/" + correction_model["correction_function"].__name__ + '_' + model + ".json", "w") as f:
     json.dump(optimized_results, f, indent=2)

    print("Optimization completed")


if __name__ == "__main__":

    # only one GCP San Francisco

    # optimize_camera_parameters(model = "RFM", correction_model={"correction_function": shift})
    optimize_camera_parameters(model = "PSM", correction_model={"correction_function": shift})
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


    # with open(f"../../San_francisco/own_GCPs/image_position.json", "r") as f:
    #     realGCPsposition = json.load(f)
    #
    # for i in range(1, 11):
    #
    #     for frame in realGCPsposition["c1"]:
    #         for GCP in realGCPsposition["c1"][frame]["GCPs"]:
    #             if GCP == str(i):
    #                 realGCPsposition["c1"][frame]["GCPs"][GCP]["control"] = 0
    #             else:
    #                 realGCPsposition["c1"][frame]["GCPs"][GCP]["control"] = 1
    #
    #     json.dump(realGCPsposition, open(f"../../San_francisco/own_GCPs/image_position.json", "w"), indent=2)
    #     # optimize_camera_parameters(method="gradient", lr=5e-9, epochs=30000, model = "PSM", correction_model=(shift_PSM, shift_initial_params_PSM))
    #     # optimize_camera_parameters(method="gradient", lr=5e-9, epochs=30000, model = "PSM", correction_model=(linear_PSM, linear_initial_params_PSM))
    #     optimize_camera_parameters(method="gradient", lr=5e-3, epochs=1000, model = "RFM", correction_model=(shift_RFM, shift_initial_params_RFM))
