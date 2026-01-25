#!/usr/bin/env python3
"""
Example: Multiple Homography Detection using Progressive-X
Converted from Jupyter notebook example_multi_homography.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
from time import time
import random
from copy import deepcopy

# Add parent directory to path to import pyprogressivex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pyprogressivex - match notebook exactly (no try/except in notebook)
import pyprogressivex

def decolorize(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

def random_color(label):
    if label == 0:
        return (255, 0, 0)
    elif label == 1:
        return (0, 255, 0)
    elif label == 2:
        return (0, 0, 255)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_labeling(img1, img2, labeling, correspondences):
    for label in range(max(labeling)):
        mask = labeling == label
        color = random_color(label)

        for i in range(len(labeling)):
            if mask[i]:
                cv2.circle(img1, (round(correspondences[i][0]), round(correspondences[i][1])), 6, color, -1)
                cv2.circle(img2, (round(correspondences[i][2]), round(correspondences[i][3])), 6, color, -1)

    plt.imshow(img1)
    plt.savefig('output_homogprahy1.png')
    plt.figure()
    plt.imshow(img2)
    plt.savefig('output_homogprahy2.png')

def verify_pyprogressivex(img1, img2, kps1, kps2, tentatives):
    correspondences = np.float32([(kps1[m.queryIdx].pt + kps2[m.trainIdx].pt) for m in tentatives]).reshape(-1, 4)
    threshold = 1.0
    
    homographies, labeling = pyprogressivex.findHomographies(
        np.ascontiguousarray(correspondences), 
        img1.shape[1], img1.shape[0], 
        img2.shape[1], img2.shape[0],
        threshold=threshold,
        conf=0.5,
        spatial_coherence_weight=0.0,
        neighborhood_ball_radius=200.0,
        maximum_tanimoto_similarity=0.4,
        max_iters=1000,
        minimum_point_number=10,
        maximum_model_number=-1,
        sampler_id=3,
        do_logging=False)    
    return homographies, labeling

def main():
    # Match notebook structure exactly - no extra print statements
    # Load the images (Cell 1)
    img1 = cv2.cvtColor(cv2.imread('img/unihouse1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('img/unihouse2.png'), cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.figure()
    plt.imshow(img2)
    
    # Detect SIFT features and match (Cell 2)
    det = cv2.SIFT_create(8000)
    kps1, descs1 = det.detectAndCompute(img1, None)
    kps2, descs2 = det.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    SNN_threshold = 0.9
    matches = bf.knnMatch(descs1, descs2, k=2)
    
    # Apply ratio test
    snn_ratios = []
    tentatives = []
    for m, n in matches:
        if m.distance < SNN_threshold * n.distance:
            tentatives.append(m)
            snn_ratios.append(m.distance / n.distance)
    
    sorted_indices = np.argsort(snn_ratios)
    tentatives = list(np.array(tentatives)[sorted_indices])
    
    # Run Progressive-X (Cell 5)
    t = time()
    homographies, labeling = verify_pyprogressivex(img1, img2, kps1, kps2, tentatives)
    model_number = homographies.size / 9
    
    print('Time = ', time() - t, ' sec')
    print('Models found = {}'.format(model_number))
    
    # Visualize results (Cell 6)
    correspondences = np.array([list(kps1[m.queryIdx].pt + kps2[m.trainIdx].pt) for m in tentatives])
    draw_labeling(img1, img2, labeling, correspondences)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
