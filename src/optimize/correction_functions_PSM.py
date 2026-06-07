import numpy as np
import torch

shift_initial_params_PSM = [torch.zeros(7, dtype=torch.float64)]
linear_initial_params_PSM = [torch.zeros(7, 7, dtype=torch.float64), torch.zeros(7, dtype=torch.float64)]
rotate_initial_params_PSM = [torch.zeros(6, dtype=torch.float64)]
quadratic_initial_params_PSM = [torch.zeros(7, 7, 7, dtype=torch.float64), torch.zeros(7, 7, dtype=torch.float64), torch.zeros(7, dtype=torch.float64)]

def linear_flatten_PSM(linear_initial_params):

    matrix = linear_initial_params[0].numpy()
    vector = linear_initial_params[1].numpy()

    return np.concatenate((matrix.flatten(), vector))

def linear_unflatten_PSM(params):

    matrix = params[:49].reshape(7, 7)
    vector = params[49:]

    return [matrix, vector]

def shift_flatten_PSM(shift_initial_params):
    return shift_initial_params[0].numpy()

def shift_unflatten_PSM(params):
    return [params]

def rotate_flatten_PSM(rotate_initial_params):
    return rotate_initial_params[0].numpy()

def rotate_unflatten_PSM(params):
    return [params]

def quadratic_flatten_PSM(quadratic_initial_params):

    tensor = quadratic_initial_params[0].numpy()
    matrix = quadratic_initial_params[1].numpy()
    vector = quadratic_initial_params[2].numpy()

    return np.concatenate((tensor.flatten(), matrix.flatten(), vector))

def quadratic_unflatten_PSM(params):
    tensor = params[:343].reshape(7, 7, 7)
    matrix = params[343:392].reshape(7, 7)
    vector = params[392:]

    return [tensor, matrix, vector]

def linear_PSM(params, quaternion, sat_position):

    matrix = params[0]
    vector = params[1]

    out = (1e-4 * matrix + torch.eye(7, dtype=torch.float64)) @ torch.cat((quaternion, sat_position)) + vector

    q = out[:4]
    sat_pos = out[4:]
    return q, sat_pos

def shift_PSM(params, quaternion, sat_position):

    out = torch.cat((quaternion, sat_position)) + params[0]

    return out[:4], out[4:]

def single_rot_PSM(i, j, theta):
    R = torch.eye(4, dtype=torch.float64)
    c, s = torch.cos(theta), torch.sin(theta)

    R[i, i] = c
    R[j, j] = c
    R[i, j] = s
    R[j, i] = -s

    return R

def rotate_PSM(params, quaternion, sat_position):

    params = params[0]

    R12 = single_rot_PSM(0, 1, params[0])
    R34 = single_rot_PSM(2, 3, params[1])
    R13 = single_rot_PSM(0, 2, params[2])
    R24 = single_rot_PSM(1, 3, params[3])
    R14 = single_rot_PSM(0, 3, params[4])
    R23 = single_rot_PSM(1, 2, params[5])

    q = R12 @ R34 @ R13 @ R24 @ R14 @ R23 @ quaternion
    sat_pos = sat_position

    return q, sat_pos

def quadratic_PSM(params, quaternion, sat_position):

    tensor = params[0]
    matrix = params[1]
    vector = params[2]
    x = torch.cat((quaternion, sat_position))

    out = 1e-12 * x.T @ tensor @ x + (1e-4 * matrix + torch.eye(7, dtype=torch.float64)) @ x + vector

    q = out[:4]
    sat_pos = out[4:]

    return q, sat_pos