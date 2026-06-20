import numpy as np
import torch

shift_initial_params_PSM = np.zeros(7, dtype=np.float64)
linear_initial_params_PSM = np.zeros(7*7 + 7, dtype=np.float64)
quadratic_initial_params_PSM = np.zeros(7*7*7 + 7*7 + 7, dtype=np.float64)

shift_initial_params_RFM = np.zeros(80, dtype=np.float64)
linear_initial_params_RFM = np.zeros(80*80 + 80, dtype=np.float64)
quadratic_initial_params_RFM = np.zeros(80*80*80 + 80*80 + 80, dtype=np.float64)


def linear(args, no_parameters, numpy=False, linear_constraint=1e-4, **kwargs):
    if numpy:
        def fun(params):
            matrix = params[:no_parameters ** 2].reshape(no_parameters, no_parameters)
            vector = params[no_parameters ** 2:]

            eye = np.eye(no_parameters, dtype=np.float64)

            return (linear_constraint * matrix + eye) @ args + vector

    else:
        def fun(params):
            matrix = params[:no_parameters ** 2].view(no_parameters, no_parameters)
            vector = params[no_parameters ** 2:]

            eye = torch.eye(no_parameters, dtype=torch.float64)

            return (linear_constraint * matrix + eye) @ args + vector

    return fun

def quadratic(args, no_parameters, numpy=False, linear_constraint=1e-4, quadratic_constraint=1e-12, **kwargs):
    if numpy:
        def fun(params):
            tensor = params[:no_parameters**3].reshape(no_parameters, no_parameters, no_parameters)
            matrix = params[no_parameters**3:no_parameters**3 + no_parameters**2].reshape(no_parameters, no_parameters)
            vector = params[no_parameters**3 + no_parameters**2:]

            eye = np.eye(no_parameters, dtype=np.float64)

            return (quadratic_constraint * np.tensordot(tensor, args, axes=([1], [0])) + linear_constraint * matrix + eye) @ args + vector

    else:
        def fun(params):
            tensor = params[:no_parameters**3].view(no_parameters, no_parameters, no_parameters)
            matrix = params[no_parameters**3:no_parameters**3 + no_parameters**2].view(no_parameters, no_parameters)
            vector = params[no_parameters**3 + no_parameters**2:]

            eye = torch.eye(no_parameters, dtype=torch.float64)

            return (quadratic_constraint * torch.tensordot(tensor, args, dims=([1], [0])) + linear_constraint * matrix + eye) @ args + vector


    return fun

def shift(args, **kwargs):
    def fun(params):
        return args + params

    return fun