import numpy as np
import torch

shift_initial_params_PSM = np.zeros(7, dtype=np.float64)
linear_initial_params_PSM = np.zeros(7*7 + 7, dtype=np.float64)
quadratic_initial_params_PSM = np.zeros(7*7*7 + 7*7 + 7, dtype=np.float64)

def linear_PSM(params, args, linear_constraint=1e-4):

    matrix = params[:49].view(7, 7)
    vector = params[49:]

    out = (linear_constraint * matrix + torch.eye(7, dtype=torch.float64)) @ args + vector

    q = out[:4]
    sat_pos = out[4:]
    return q, sat_pos

def shift_PSM(params, args):

    out = args + params

    return out[:4], out[4:]

def quadratic_PSM(params, args, linear_constraint=1e-4, quadratic_constraint=1e-12):
    tensor = params[:343].view(7, 7, 7)
    matrix = params[343:392].view(7, 7)
    vector = params[392:]

    out = quadratic_constraint * args.T @ tensor @ args + (linear_constraint * matrix + torch.eye(7, dtype=torch.float64)) @ args + vector

    q = out[:4]
    sat_pos = out[4:]

    return q, sat_pos