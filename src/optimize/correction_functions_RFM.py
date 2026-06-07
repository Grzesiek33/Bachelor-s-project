import numpy as np
import torch

shift_initial_params_RFM = [torch.zeros(80, dtype=torch.float64)]
linear_initial_params_RFM = [torch.eye(80, dtype=torch.float64), torch.zeros(80, dtype=torch.float64)]
quadratic_initial_params_RFM = [torch.zeros(80,80,80, dtype=torch.float64), torch.eye(80, dtype=torch.float64), torch.zeros(80, dtype=torch.float64)]

def linear_flatten_RFM(linear_initial_params):

    matrix = linear_initial_params[0].numpy()
    vector = linear_initial_params[1].numpy()

    return np.concatenate((matrix.flatten(), vector))

def linear_unflatten_RFM(params):

    matrix = params[:6400].reshape(80, 80)
    vector = params[6400:]

    return [matrix, vector]

def shift_flatten_RFM(shift_initial_params):
    return shift_initial_params[0].numpy()

def shift_unflatten_RFM(params):
    return [params]

def quadratic_flatten_RFM(quadratic_initial_params):

    tensor = quadratic_initial_params[0].numpy()
    matrix = quadratic_initial_params[1].numpy()
    vector = quadratic_initial_params[2].numpy()

    return np.concatenate((tensor.flatten(), matrix.flatten(), vector))

def quadratic_unflatten_RFM(params):
    tensor = params[:512_000].reshape(80, 80, 80)
    matrix = params[512_000:512_000+6400].reshape(80, 80)
    vector = params[512_000+6400:]

    return [tensor, matrix, vector]

def linear_RFM(params, Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs):

    matrix = params[0]
    vector = params[1]

    out = matrix @ torch.cat((Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs)) + vector

    line_num = out[:20]
    line_den = out[20:40]
    samp_num = out[40:60]
    samp_den = out[60:]
    return line_num, line_den, samp_num, samp_den

def shift_RFM(params, Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs):

    out = torch.cat((Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs)) + params[0]

    line_num = out[:20]
    line_den = out[20:40]
    samp_num = out[40:60]
    samp_den = out[60:]
    return line_num, line_den, samp_num, samp_den

def quadratic_RFM(params, Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs):

    tensor = params[0]
    matrix = params[1]
    vector = params[2]
    x = torch.cat((Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs))

    out = x.T @ tensor @ x + matrix @ x + vector

    line_num = out[:20]
    line_den = out[20:40]
    samp_num = out[40:60]
    samp_den = out[60:]
    return line_num, line_den, samp_num, samp_den