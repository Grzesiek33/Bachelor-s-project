import torch
import numpy


def RFM_torch(lat_off, lat_scale, long_off, long_scale, height_off, height_scale, line_off, line_scale, samp_off, samp_scale, Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs):

    def model(B, L, H):
        B = (B - lat_off) / lat_scale
        L = (L - long_off) / long_scale
        H = (H - height_off) / height_scale

        args = torch.tensor([1,
    L,
    B,
    H,
    B*L,
    L*H,
    B*H,
    L*L,
    B*B,
    H*H,
    B*L*H,
    L*L*L,
    B*B*L,
    L*H*H,
    B*L*L,
    B*B*B,
    B*H*H,
    L*L*H,
    B*B*H,
    H*H*H], dtype=torch.float64)
        l = (Line_num_coeffs @ args) / (Line_den_coeffs @ args)
        s = (Samp_num_coeffs @ args) / (Samp_den_coeffs @ args)
        return l * line_scale + line_off, s * samp_scale + samp_off

    return model


def RFM_numpy(lat_off, lat_scale, long_off, long_scale, height_off, height_scale, line_off, line_scale, samp_off, samp_scale, Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs):

    def model(B, L, H):
        B = (B - lat_off) / lat_scale
        L = (L - long_off) / long_scale
        H = (H - height_off) / height_scale

        args = numpy.array([1,
    L,
    B,
    H,
    B*L,
    L*H,
    B*H,
    L*L,
    B*B,
    H*H,
    B*L*H,
    L*L*L,
    B*B*L,
    L*H*H,
    B*L*L,
    B*B*B,
    B*H*H,
    L*L*H,
    B*B*H,
    H*H*H], dtype=numpy.float64)
        l = (Line_num_coeffs @ args) / (Line_den_coeffs @ args)
        s = (Samp_num_coeffs @ args) / (Samp_den_coeffs @ args)
        return l * line_scale + line_off, s * samp_scale + samp_off

    return model