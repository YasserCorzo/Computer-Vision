import cv2
import matplotlib.pyplot as plt
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
#print(F)
#displayEpipolarF(img1, img2, F)

# retrieve intrinsics
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# calculate essential matrix
#E = essentialMatrix(F, K1, K2)
#print(E)
#np.savez('q3_1', E)

# testing Q4.1
epipolarMatchGUI(img1, img2, F)
