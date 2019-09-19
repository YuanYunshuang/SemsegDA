#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) with pytorch
"""
import numpy as np
import torch


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    :param mu1: Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    :param sigma1: The covariance matrix over activations for generated samples.
    :param mu2: The sample mean over activations, precalculated on an
               representative data set.
    :param sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    :param eps:
    :return: The Frechet Distance.
    """

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = torch.sqrt(sigma1.dot(sigma2))
    if not torch.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = torch.sqrt((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean.numpy()):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = torch.trace(covmean)

    return (diff.dot(diff) + torch.trace(sigma1) +
            torch.trace(sigma2) - 2 * tr_covmean)


def calculate_ditribution(input):
    mu = torch.mean(input)
    sigma = torch.cov(input)
    return mu, sigma


def calculate_fid(input1, input2):
    mu1, sigma1 = calculate_ditribution(input1)
    mu2, sigma2 = calculate_ditribution(input2)

    return frechet_distance(mu1, sigma1, mu2, sigma2)
