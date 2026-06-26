import numpy as np
import torch

def zero_based_initial_params(correction_function, no_parameters):
    if correction_function == "shift":
        return np.zeros(no_parameters, dtype=np.float64)
    elif correction_function == "linear":
        return np.zeros(no_parameters**2 + no_parameters, dtype=np.float64)
    elif correction_function == "quadratic":
        return np.zeros(no_parameters**3 + no_parameters**2 + no_parameters, dtype=np.float64)
    elif correction_function == "rotation":
        return np.array([1] + [0 for i in range(no_parameters-1)], dtype=np.float64)
    else:
        raise ValueError(f"Unknown initial parameters for correction function: {correction_function}")

def linear(args, no_parameters, device=torch.device("cpu"), numpy=False, linear_constraint=1e-4, **kwargs):
    if numpy:

        eye = np.eye(no_parameters, dtype=np.float64)

        def fun(params):
            matrix = params[:no_parameters ** 2].reshape(no_parameters, no_parameters)
            vector = params[no_parameters ** 2:]

            return (linear_constraint * matrix + eye) @ args + vector

    else:
        eye = torch.eye(no_parameters, dtype=torch.float64, device=device)

        def fun(params):
            matrix = params[:no_parameters ** 2].view(no_parameters, no_parameters)
            vector = params[no_parameters ** 2:]

            return (linear_constraint * matrix + eye) @ args + vector

    return fun

def quadratic(args, no_parameters, device=torch.device("cpu"), numpy=False, linear_constraint=1e-4, quadratic_constraint=1e-12, **kwargs):
    if numpy:

        eye = np.eye(no_parameters, dtype=np.float64)

        def fun(params):
            tensor = params[:no_parameters**3].reshape(no_parameters, no_parameters, no_parameters)
            matrix = params[no_parameters**3:no_parameters**3 + no_parameters**2].reshape(no_parameters, no_parameters)
            vector = params[no_parameters**3 + no_parameters**2:]

            return (quadratic_constraint * np.tensordot(tensor, args, axes=([1], [0])) + linear_constraint * matrix + eye) @ args + vector

    else:

        eye = torch.eye(no_parameters, dtype=torch.float64, device=device)

        def fun(params):
            tensor = params[:no_parameters**3].view(no_parameters, no_parameters, no_parameters)
            matrix = params[no_parameters**3:no_parameters**3 + no_parameters**2].view(no_parameters, no_parameters)
            vector = params[no_parameters**3 + no_parameters**2:]

            return (quadratic_constraint * torch.tensordot(tensor, args, dims=([1], [0])) + linear_constraint * matrix + eye) @ args + vector


    return fun

def shift(args, **kwargs):
    def fun(params):
        return args + params

    return fun

def rotation(args, numpy=False, **kwargs):
    if numpy:

        w1, x1, y1, z1 = args[:4]

        def fun(params):
            w2, x2, y2, z2 = params

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return np.array([w, x, y, z, *args[4:]], dtype=np.float64)

    else:
        w1, x1, y1, z1 = args[..., 0], args[..., 1], args[..., 2], args[..., 3]

        def fun(params):

            w2, x2, y2, z2 = params[..., 0], params[..., 1], params[..., 2], params[..., 3]

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return torch.stack([w, x, y, z, *args[..., 4:]], dim=-1)

    return fun