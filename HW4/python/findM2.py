'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import cv2
import numpy as np

from helper import *
from submission import *

# retrieve images
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')

# retrieve correspondences
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']

# calculate M
M = max(img1.shape[0], img1.shape[1])
F = eightpoint(pts1, pts2, M)

# retrieve intrinsics
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# calculate essential matrix
E = essentialMatrix(F, K1, K2)

# calculate projection matrix 1
C1 = K1 @ np.hstack((np.eye(3), np.zeros(3).reshape(-1, 1)))

# retrieve the M2s (4 possible extrinsics)
M2s = camera2(E)

# triangulate with all possible projection camera matrix 2
w1, err1 = triangulate(C1, pts1, K2 @ M2s[:, :, 0], pts2)
w2, err2 = triangulate(C1, pts1, K2 @ M2s[:, :, 1], pts2)
w3, err3 = triangulate(C1, pts1, K2 @ M2s[:, :, 2], pts2)
w4, err4 = triangulate(C1, pts1, K2 @ M2s[:, :, 3], pts2)

print(err1, err2, err3, err4)

# return M2 that has lowest triangulate error
l = [(err1, 0), (err2, 1), (err3, 2), (err4, 3)]

M2 = M2s[:, :, min(l)[1]]
np.savez('q3_3', M2)


