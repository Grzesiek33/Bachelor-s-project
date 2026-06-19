import numpy
import numpy as np
import torch

shift_initial_params_RFM = numpy.zeros(80, dtype=np.float64)
linear_initial_params_RFM = numpy.zeros(80*80 + 80, dtype=np.float64)
quadratic_initial_params_RFM = numpy.zeros(80*80*80 + 80*80 + 80, dtype=np.float64)

def linear_RFM(params, args):
    matrix = params[:6400].view(80, 80)
    vector = params[6400:]

    out = matrix @ args + vector

    line_num = out[:20]
    line_den = out[20:40]
    samp_num = out[40:60]
    samp_den = out[60:]
    return line_num, line_den, samp_num, samp_den

def shift_RFM(params, args):

    out = args + params

    line_num = out[:20]
    line_den = out[20:40]
    samp_num = out[40:60]
    samp_den = out[60:]
    return line_num, line_den, samp_num, samp_den

def quadratic_RFM(params, args):

    tensor = params[:512_000].reshape(80, 80, 80)
    matrix = params[512_000:512_000 + 6400].reshape(80, 80)
    vector = params[512_000 + 6400:]

    out = args.T @ tensor @ args + matrix @ args + vector

    line_num = out[:20]
    line_den = out[20:40]
    samp_num = out[40:60]
    samp_den = out[60:]
    return line_num, line_den, samp_num, samp_den