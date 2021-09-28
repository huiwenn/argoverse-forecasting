from typing import List, Set, Dict, Tuple, Optional, Union
from scipy import special
import numpy as np
import torch
import json
import os


def to5d(x: torch.Tensor):
    batch_size = x.shape[0]

    # INPUT to lstm = [x, y , log(sigmax), log(sigmay), tanh^{-1}(rho)]
    #new_input = torch.cat([x[:, :2], -5 * torch.ones(batch_size, 2), torch.zeros(batch_size, 1)], dim=1)
    new_input = torch.cat([x[:, :2], torch.zeros(batch_size, 3)], dim=1)

    return new_input


def Gaussian2d(x: torch.Tensor) -> torch.Tensor:
    """Computes the parameters of a bivariate 2D Gaussian."""
    x_mean = x[:, 0]
    y_mean = x[:, 1]
    sigma_x = torch.exp(x[:, 2])  # not inverse, see if it works
    sigma_y = torch.exp(x[:, 3])  # not inverse
    rho = torch.tanh(x[:, 4])
    return torch.stack([x_mean, y_mean, sigma_x, sigma_y, rho], dim=1)


def nll_loss(pred: torch.Tensor, data: torch.Tensor, scale=1) -> torch.Tensor:
    """Negative log loss for single-variate gaussian, can probably be faster"""
    x_mean = pred[:, 0]
    y_mean = pred[:, 1]
    x_delta = x_mean - data[:, 0]
    y_delta = y_mean - data[:, 1]
    x_sigma = pred[:, 2]
    y_sigma = pred[:, 3]
    rho = pred[:, 4]
    root_det_epsilon = torch.pow(1-torch.pow(rho,2), 0.5) * x_sigma * y_sigma
    loss = torch.log(2*3.14159*root_det_epsilon) \
            + 0.5 * torch.pow(root_det_epsilon, -2) \
                * (torch.pow(x_sigma, 2) * torch.pow(y_delta, 2) \
                + torch.pow(y_sigma, 2) * torch.pow(x_delta, 2) \
                - 2 * rho * x_sigma * y_sigma * x_delta * y_delta)

    return torch.mean(loss)


def mis_loss(pred: torch.Tensor, data: torch.Tensor, alpha=0.9, scale=1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean interval score for single-variate guassian"""
    x_mean = pred[:, 0]
    y_mean = pred[:, 1]
    x_delta = (x_mean - data[:, 0])
    y_delta = (y_mean - data[:, 1])
    x_sigma = pred[:, 2]
    y_sigma = pred[:, 3]
    rho = pred[:, 4]

    ohr = torch.pow(1 - torch.pow(rho, 2), 0.5)
    root_det_epsilon = ohr * x_sigma * y_sigma
    c_alpha = - 2 * np.log(1 - alpha)

    c_ = (torch.pow(x_sigma, 2) * torch.pow(y_delta, 2) \
          + torch.pow(y_sigma, 2) * torch.pow(x_delta, 2) \
          - 2 * rho * x_sigma * y_sigma * x_delta * y_delta) * torch.pow(root_det_epsilon, -2)  # c prime

    c_delta = c_ - c_alpha
    c_delta = torch.where(c_delta > 0, c_delta, torch.zeros_like(c_delta))

    mrs = root_det_epsilon * (c_alpha + scale * c_delta / alpha)
    return torch.mean(mrs)