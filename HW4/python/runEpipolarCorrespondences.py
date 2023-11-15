import cv2
import numpy as np

from helper import *
from submission import *

# testing Q2.1
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']

M = max(img1.shape[0], img1.shape[1])
F = eightpoint(pts1, pts2, M)

epipolarMatchGUI(img1, img2, F)
