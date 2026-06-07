import json
import numpy as np
import torch


def create_extrinsic(quaternion, sat_position):
    qw, qx, qy, qz = quaternion
    sat_x, sat_y, sat_z = sat_position

    z = torch.zeros_like(qw)
    o = torch.ones_like(qw)

    r0 = torch.stack([1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, z])
    r1 = torch.stack([2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw, z])
    r2 = torch.stack([2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy, z])
    r3 = torch.stack([z, z, z, o])
    R = torch.stack([r0, r1, r2, r3])

    p0 = torch.stack([o, z, z, -sat_x])
    p1 = torch.stack([z, o, z, -sat_y])
    p2 = torch.stack([z, z, o, -sat_z])
    p3 = torch.stack([z, z, z, o])
    pos = torch.stack([p0, p1, p2, p3])

    return R @ pos