import json
import numpy as np
import torch

def PSM(P_camera, P_intrinsic, quaternion, sat_position, device, numpy=False):

    if numpy:
        def model(x_ecef, y_ecef, z_ecef):

            P_extrinsic = create_extrinsic(quaternion, sat_position, numpy=True, device=device)

            im_space = P_camera @ P_intrinsic @ P_extrinsic @ np.array([x_ecef, y_ecef, z_ecef, 1], dtype=np.float64)
            return im_space[0] / im_space[2], im_space[1] / im_space[2]
    else:
        def model(x_ecef, y_ecef, z_ecef):
            P_extrinsic = create_extrinsic(quaternion, sat_position, numpy=False, device=device)
            im_space = P_camera @ P_intrinsic @ P_extrinsic @ torch.tensor([x_ecef, y_ecef, z_ecef, 1], dtype=torch.float64)
            return im_space[0] / im_space[2], im_space[1] / im_space[2]

    return model

def create_extrinsic(quaternion, sat_position, device, numpy=False):
    qw, qx, qy, qz = quaternion
    sat_x, sat_y, sat_z = sat_position

    if numpy:
        r0 = np.array([1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw, 0], dtype=np.float64)
        r1 = np.array([2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw, 0], dtype=np.float64)
        r2 = np.array([2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy, 0], dtype=np.float64)
        r3 = np.array([0, 0, 0, 1], dtype=np.float64)
        R = np.stack([r0, r1, r2, r3], dtype=np.float64)

        p0 = np.array([1, 0, 0, -sat_x], dtype=np.float64)
        p1 = np.array([0, 1, 0, -sat_y], dtype=np.float64)
        p2 = np.array([0, 0, 1, -sat_z], dtype=np.float64)
        p3 = np.array([0, 0, 0, 1], dtype=np.float64)
        pos = np.stack([p0, p1, p2, p3], dtype=np.float64)
    else:

        z = torch.zeros_like(qw, dtype=torch.float64, device=device)
        o = torch.ones_like(qw, dtype=torch.float64, device=device)

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
