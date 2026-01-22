import torch
import cv2
import numpy as np

def simple_gradcam(image, gradients):
    heatmap = np.mean(gradients, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap
