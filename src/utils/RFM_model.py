import torch


def RFM(lat_off, lat_scale, long_off, long_scale, height_off, height_scale, line_off, line_scale, samp_off, samp_scale, Line_num_coeffs, Line_den_coeffs, Samp_num_coeffs, Samp_den_coeffs):

    def model(P, L, H):

        P = (P - lat_off) / lat_scale
        L = (L - long_off) / long_scale
        H = (H - height_off) / height_scale

        args = torch.tensor([1,
    L,
    P,
    H,
    P*L,
    L*H,
    P*H,
    L*L,
    P*P,
    H*H,
    P*L*H,
    L*L*L,
    P*P*L,
    L*H*H,
    P*L*L,
    P*P*P,
    P*H*H,
    L*L*H,
    P*P*H,
    H*H*H], dtype=torch.float64)
        l = (Line_num_coeffs @ args) / (Line_den_coeffs @ args)
        s = (Samp_num_coeffs @ args) / (Samp_den_coeffs @ args)
        return l * line_scale + line_off, s * samp_scale + samp_off

    return model