import cv2
import numpy as np

from helper import *
from submission import *

# retrieve images
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')

# retrieve correspondences
data = np.load('../data/some_corresp_noisy.npz')
pts1 = data['pts1']
pts2 = data['pts2']

# calculate M
M = max(img1.shape[0], img1.shape[1])

# calculate F (without RANSAC)
F = eightpoint(pts1, pts2, M)
print(F)

# calculate F with ransac
F_ransac, inliers = ransacF(pts1, pts2, M, tol=0.67)
print(F_ransac)
print(inliers.sum())