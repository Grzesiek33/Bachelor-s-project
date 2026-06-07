import torch


def MSE(real_im_x, real_im_y, im_x, im_y):
    return (im_x - real_im_x) ** 2 + (im_y - real_im_y) ** 2
