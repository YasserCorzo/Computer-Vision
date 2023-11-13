'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np

from helper import *
from submission import *
'''
# testing Q2.1
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']

M = max(img1.shape[0], img1.shape[1])
F = eightpoint(pts1, pts2, M)
#displayEpipolarF(img1, img2, F)

# testing Q4.1
epipolarMatchGUI(img1, img2, F)
'''
# load the two temple images and their corresponding points
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']

# run eight point to compute F
M = max(img1.shape[0], img1.shape[1])
F = eightpoint(pts1, pts2, M)

# load 288 hand-selected points from im1 
temple_data = np.load('../data/templeCoords.npz')
x1 = temple_data['x1']
y1 = temple_data['y1']
temple_pts1 = np.hstack((x1, y1))

# find epipolar correspondences
temple_pts2 = []
for i in range(len(temple_pts1)):
    x1_i, y1_i = temple_pts1[i, :]
    x2_i, y2_i = epipolarCorrespondence(img1, img2, F, x1_i, y1_i)
    temple_pts2.append([x2_i, y2_i])

temple_pts2 = np.array(temple_pts2)

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
w1, err1 = triangulate(C1, temple_pts1, K2 @ M2s[:, :, 0], temple_pts2)
w2, err2 = triangulate(C1, temple_pts1, K2 @ M2s[:, :, 1], temple_pts2)
w3, err3 = triangulate(C1, temple_pts1, K2 @ M2s[:, :, 2], temple_pts2)
w4, err4 = triangulate(C1, temple_pts1, K2 @ M2s[:, :, 3], temple_pts2)

# return 3D coordinates with corresponding M2 that has lowest triangulate error 
# and where projected 3D z-coordinates is positive
l = [(err1, 0), (err2, 1), (err3, 2), (err4, 3)]
ws = [w1, w2, w3, w4]
w = None
min_error = float('inf')
for (err, i) in l:
    w_i = ws[i]
    if np.all(w_i[:, -1] > 0):
        min_error = err
        w = w_i
        
# scatter plot temple 3D coordinates
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(w[:, 0], w[:, 1], w[:, 2])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()    
    



