#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 12:03:53 2026

@author: sydneydolan
"""

import numpy as np



def generate_pentagram(
    points_per_edge=25,
    radius=1.0,
    noise_sigma=0.01,
    jitter=0.1,
    seed=0
):
    np.random.seed(seed)

    angles = 2 * np.pi * np.arange(5) / 5
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius

    edges = [(0,2), (2,4), (4,1), (1,3), (3,0)]

    points = []
    labels = []

    for i, (a, b) in enumerate(edges):
        v0, v1 = vertices[a], vertices[b]

        # Stratified sampling in [0,1]
        bins = np.linspace(0.0, 1.0, points_per_edge + 1)
        t = bins[:-1] + np.random.rand(points_per_edge) * (bins[1] - bins[0]) * jitter

        segment = (1 - t)[:, None] * v0 + t[:, None] * v1
        segment += np.random.normal(scale=noise_sigma, size=segment.shape)

        points.append(segment)
        labels += [i] * points_per_edge

    return np.vstack(points), np.array(labels)



data,labels = generate_pentagram()

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(data[:,0],data[:,1],s=4)
