import numpy as np
import torch

def zero_based_initial_params(correction_function, no_parameters):
    if correction_function == "shift":
        return np.zeros(no_parameters, dtype=np.float64)
    elif correction_function == "linear":
        return np.zeros(no_parameters**2 + no_parameters, dtype=np.float64)
    elif correction_function == "quadratic":
        return np.zeros(no_parameters**3 + no_parameters**2 + no_parameters, dtype=np.float64)
    else:
        raise ValueError(f"Unknown initial parameters for correction function: {correction_function}")

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