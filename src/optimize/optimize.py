import os

import numpy as np
from scipy.optimize import minimize
import json

from shapely import wkt

from src.utils.create_extrinsic import create_extrinsic
from correction_functions_PSM import linear_PSM, linear_initial_params_PSM, shift_PSM, shift_initial_params_PSM, rotate_PSM, rotate_initial_params_PSM, quadratic_PSM, quadratic_initial_params_PSM, linear_flatten_PSM, linear_unflatten_PSM, shift_flatten_PSM, shift_unflatten_PSM, rotate_flatten_PSM, rotate_unflatten_PSM, quadratic_flatten_PSM, quadratic_unflatten_PSM
from optimization_function import MSE
from correction_functions_RFM import linear_flatten_RFM, linear_unflatten_RFM, shift_flatten_RFM, shift_unflatten_RFM, quadratic_flatten_RFM, quadratic_unflatten_RFM, linear_RFM, shift_RFM, quadratic_RFM, linear_initial_params_RFM, shift_initial_params_RFM, quadratic_initial_params_RFM
import re
import torch
from src.utils.RFM_model import RFM

def helper_function_PSM(params, frame, GCP_meta_data, Frameinfo, correction_function=linear_PSM, optimization_function=MSE, q_constraint : float = 1, unflatten = linear_unflatten_PSM, nump : bool = False):

    if nump:
        params = unflatten(params)
        params = [torch.tensor(p, dtype=torch.float64) for p in params]

    P_camera = torch.tensor(Frameinfo["P_camera"], dtype=torch.float64)
    P_intrinsic = torch.tensor(Frameinfo["P_intrinsic"], dtype=torch.float64)

    exterior_rotation = Frameinfo["exterior_orientation"]

    quaternion = torch.tensor([exterior_rotation["qw_ecef"], exterior_rotation["qx_ecef"], exterior_rotation["qy_ecef"],
                           exterior_rotation["qz_ecef"]], dtype=torch.float64)

    sat_position = torch.tensor(
        [exterior_rotation["x_ecef_meters"], exterior_rotation["y_ecef_meters"], exterior_rotation["z_ecef_meters"]], dtype=torch.float64)

    q, sat_pos = correction_function(params, quaternion, sat_position)

    P_extrinsic = create_extrinsic(q, sat_pos)

    x_ecef = float(GCP_meta_data["x_ecef"])
    y_ecef = float(GCP_meta_data["y_ecef"])
    z_ecef = float(GCP_meta_data["z_ecef"])

    row = frame["row"]
    col = frame["col"]

    total_error = torch.tensor(0, dtype=torch.float64)

    im_space = P_camera @ P_intrinsic @ P_extrinsic @ torch.tensor([x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64)
    im_x = im_space[0] / im_space[2]
    im_y = im_space[1] / im_space[2]

    error = optimization_function(col, row, im_x, im_y)
    total_error = total_error + error
    total_error = total_error + q_constraint * ((1 - ( q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 )) ** 2)

    if nump:
        return total_error.item()
    return total_error

def helper_function_RFM(params, frame, GCP_meta_data, Frameinfo, correction_function=linear_RFM, optimization_function=MSE, unflatten = linear_unflatten_RFM, nump : bool = False):

    if nump:
        params = unflatten(params)
        params = [torch.tensor(p, dtype=torch.float64) for p in params]

    Line_num_coeffs = torch.tensor(Frameinfo[f"LINE_NUM_COEFFS"], dtype=torch.float64)
    Line_den_coeffs = torch.tensor(Frameinfo[f"LINE_DEN_COEFFS"], dtype=torch.float64)
    Samp_num_coeffs = torch.tensor(Frameinfo[f"SAMP_NUM_COEFFS"], dtype=torch.float64)
    Samp_den_coeffs = torch.tensor(Frameinfo[f"SAMP_DEN_COEFFS"], dtype=torch.float64)

    line_off = Frameinfo["LINE_OFF"]
    samp_off = Frameinfo["SAMP_OFF"]
    lat_off = Frameinfo["LAT_OFF"]
    long_off = Frameinfo["LONG_OFF"]
    height_off = Frameinfo["HEIGHT_OFF"]
    line_scale = Frameinfo["LINE_SCALE"]
    samp_scale = Frameinfo["SAMP_SCALE"]
    lat_scale = Frameinfo["LAT_SCALE"]
    long_scale = Frameinfo["LONG_SCALE"]
    height_scale = Frameinfo["HEIGHT_SCALE"]

    Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs = correction_function(params, Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs)

    P = float(GCP_meta_data["lat"])
    L = float(GCP_meta_data["lon"])
    H = float(GCP_meta_data["alt"])

    row = frame["row"]
    col = frame["col"]

    total_error = torch.tensor(0, dtype=torch.float64)

    model = RFM(lat_off, lat_scale, long_off, long_scale, height_off, height_scale, line_off, line_scale, samp_off, samp_scale, Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs)

    im_y, im_x = model(L, P, H)

    error = optimization_function(col, row, im_x, im_y)
    total_error = total_error + error

    if nump:
        return total_error.item()
    return total_error

def optimize_camera_parameters(correction_model = (linear_PSM, linear_initial_params_PSM, linear_flatten_PSM, linear_unflatten_PSM), correction_for: str = "c1", restricted_to: list = None, exclude: list = None, method: str = 'Nelder-Mead', lr=1e-5, epochs=1000, optimization_function=MSE, q_constraint : float = 1, model ="PSM",
                               cities=None):
    if cities is None:
        cities = ["San_francisco", "Angkor_wat", "Cocabamba"]

    GCPinfo = {}
    realGCPsposition = {}

    for city in cities:
        with open(f"../../{city}/own_GCPs/GCPs.json", "r") as f:
            GCPinfo[city] = json.load(f)

        with open(f"../../{city}/own_GCPs/image_position.json", "r") as f:
            realGCPsposition[city] = json.load(f)

    control_GCPs = {}

    for city in cities:
        control_GCPs[city] = 0
        for cam in realGCPsposition[city]:
            for frame in realGCPsposition[city][cam]:
                for GCP in realGCPsposition[city][cam][frame]["GCPs"]:
                    if realGCPsposition[city][cam][frame]["GCPs"][GCP]["control"] == 1:
                        control_GCPs[city] += 1

    if correction_for[0] == "c":
        if model == "PSM":
            def objective_function(params):
                total_error = torch.tensor(0, dtype=torch.float64)
                for city in cities:
                    for frame in realGCPsposition[city][correction_for]:
                        with open(f"../../{city}/l1a_frames/" + frame + "_pinhole.json", "r") as f:
                            Frameinfo = json.load(f)

                        for GCP in realGCPsposition[city][correction_for][frame]["GCPs"]:
                            if realGCPsposition[city][correction_for][frame]["GCPs"][GCP]["control"] == 0 and (restricted_to is None or GCP in restricted_to) and (exclude is None or GCP not in exclude):
                                total_error = total_error + helper_function_PSM(params, realGCPsposition[city][correction_for][frame]["GCPs"][GCP], GCPinfo[city][GCP], Frameinfo, correction_function=correction_model[0], optimization_function=optimization_function, q_constraint=q_constraint, unflatten=correction_model[3], nump=(method!="gradient"))
                return total_error
        elif model == "RFM":
            def objective_function(params):

                total_error = torch.tensor(0, dtype=torch.float64)
                for city in cities:
                    with open(f"../../{city}/frameRPC.json", "r") as f:
                        Frame = json.load(f)

                    for frame in realGCPsposition[city][correction_for]:
                        Frameinfo = Frame[frame]

                        for GCP in realGCPsposition[city][correction_for][frame]["GCPs"]:
                            if realGCPsposition[city][correction_for][frame]["GCPs"][GCP]["control"] == 0 and (restricted_to is None or GCP in restricted_to) and (exclude is None or GCP not in exclude):
                                total_error = total_error + helper_function_RFM(params, realGCPsposition[city][correction_for][frame]["GCPs"][GCP], GCPinfo[city][GCP], Frameinfo, correction_function=correction_model[0], optimization_function=optimization_function, unflatten=correction_model[3], nump=(method!="gradient"))
                return total_error
        else:
            raise ValueError(f"model {model} not implemented")

    elif "." in correction_for:
        if model == "PSM":
            def objective_function(params):
                total_error = torch.tensor(0, dtype=torch.float64)
                with open(f"../../{city}/l1a_frames/" + correction_for + "_pinhole.json", "r") as f:
                    Frameinfo = json.load(f)

                cam = re.search(r"c[1,2,3]", correction_for).group(0)

                for GCP in realGCPsposition[city][cam][correction_for]["GCPs"]:
                    if realGCPsposition[city][cam][correction_for]["GCPs"][GCP]["control"] == 0 and (restricted_to is None or GCP in restricted_to) and (exclude is None or GCP not in exclude):
                        total_error = total_error + helper_function_PSM(params, realGCPsposition[city][cam][correction_for]["GCPs"][GCP], GCPinfo[city][GCP], Frameinfo, correction_function=correction_model[0], optimization_function=optimization_function, q_constraint=q_constraint, unflatten=correction_model[3], nump=(method!="gradient"))
                return total_error
        elif model == "RFM":
            def objective_function(params):
                total_error = torch.tensor(0, dtype=torch.float64)
                with open(f"../../{city}/frameRPC.json", "r") as f:
                    Frameinfo = json.load(f)[correction_for]

                cam = re.search(r"c[1,2,3]", correction_for).group(0)

                for GCP in realGCPsposition[city][cam][correction_for]["GCPs"]:
                    if realGCPsposition[city][cam][correction_for]["GCPs"][GCP]["control"] == 0 and (restricted_to is None or GCP in restricted_to) and (exclude is None or GCP not in exclude):
                        total_error = total_error + helper_function_RFM(params, realGCPsposition[city][cam][correction_for]["GCPs"][GCP], GCPinfo[city][GCP], Frameinfo, correction_function=correction_model[0], optimization_function=optimization_function, unflatten=correction_model[3], nump=(method!="gradient"))
                return total_error

    else:
        raise ValueError(f"correction for {correction_for} not implemented")

    if method != "gradient":

        result = minimize(objective_function, np.array(correction_model[2](correction_model[1])), method=method)

        optimized_params = [p.tolist() for p in correction_model[3](result.x)]

    else:
        params = [torch.nn.Parameter(p.detach().clone()) for p in correction_model[1]]

        optimizer = torch.optim.Adam(params, lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = objective_function(params)
            loss.backward()
            optimizer.step()

        optimized_params = [p.tolist() for p in params]

    if not os.path.exists("../../optimization/" + correction_model[0].__name__ + ".json"):
        with open("../../optimization/" + correction_model[0].__name__ +  ".json", "w") as f:
            f.write("{}")

    with open("../../optimization/" + correction_model[0].__name__ + ".json", "r") as f:
        optimized_results = json.load(f)
    if correction_for not in optimized_results:
        optimized_results[correction_for] = {}

    if method not in optimized_results[correction_for]:
        optimized_results[correction_for][method] = {}
    if str(cities) not in optimized_results[correction_for][method]:
        optimized_results[correction_for][method][str(cities)] = {}
    if str(control_GCPs) not in optimized_results[correction_for][method][str(cities)]:
        optimized_results[correction_for][method][str(cities)][str(control_GCPs)] = {}

    optimized_results[correction_for][method][str(cities)][str(control_GCPs)][("[]" if exclude is None else "e"+str(exclude)) if restricted_to is None else "r"+str(restricted_to)] = optimized_params

    with open("../../optimization/" + correction_model[0].__name__ + ".json", "w") as f:
     json.dump(optimized_results, f, indent=2)


if __name__ == "__main__":

    # only one GCP San Francisco

    # optimize_camera_parameters(method="gradient", lr=5e-5, epochs=3000, model = "RFM", correction_model=(shift_RFM, shift_initial_params_RFM, shift_flatten_RFM, shift_unflatten_RFM))
    # optimize_camera_parameters(method="gradient", lr=5e-9, epochs=30000, model = "PSM", correction_model=(shift_PSM, shift_initial_params_PSM, shift_flatten_PSM, shift_unflatten_PSM))

    # optimize_camera_parameters(method="gradient", lr=5e-5, epochs=3000, model = "RFM", correction_model=(linear_RFM, linear_initial_params_RFM, linear_flatten_RFM, linear_unflatten_RFM))
    # optimize_camera_parameters(method="gradient", lr=1e-11, epochs=10000, model = "PSM", correction_model=(linear_PSM, linear_initial_params_PSM, linear_flatten_PSM, linear_unflatten_PSM))

    # optimize_camera_parameters(method="gradient", lr=5e-5, epochs=3000, model = "RFM", correction_model=(quadratic_RFM, quadratic_initial_params_RFM, quadratic_flatten_RFM, quadratic_unflatten_RFM))
    optimize_camera_parameters(method="gradient", lr=1e-12, epochs=60000, model = "PSM", correction_model=(quadratic_PSM, quadratic_initial_params_PSM, quadratic_flatten_PSM, quadratic_unflatten_PSM))
