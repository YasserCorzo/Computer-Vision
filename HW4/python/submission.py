"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np

from util import *

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    
    '''
    Input:  pts1 (scaled by factor M), Nx2 matrix
            pts2 (scaled by factor M), Nx2 matrix
    Output: U, correspondences matrix, Nx9 matrix
    '''
    def compute_U(pts1, pts2):
        U = []
        for i in range(len(pts1)):
            x_i, y_i = pts1[i, 0], pts1[i, 1]
            x_i_prime, y_i_prime = pts2[i, 0], pts2[i, 1]
            U.append([x_i_prime*x_i, x_i_prime*y_i, x_i_prime, y_i_prime*x_i, y_i_prime*y_i, y_i_prime, x_i, y_i, 1])
            
        U = np.array(U)
        return U
    
    '''
    Input:  U, correspondences matrix, Nx9
    Output: F, fundamental matrix, 3x3
    '''
    def compute_F(U):
        u, s, vh = np.linalg.svd(U, full_matrices=True, compute_uv=True, hermitian=False)
        f = vh[-1]
        F = f.reshape(3, 3)
        return F 
    
    '''
    Input:  F, fundamental matrix w/o rank 2 constraint, 3x3 matrix
    Output: F_rank_2, true fundamental matrix with rank 2, 3x3 matrix
    '''
    def enforce_rank_2_constraint(F):
        u, s, vh = np.linalg.svd(F, full_matrices=True, compute_uv=True, hermitian=False)
        s[-1] = 0
        F_rank_2 = (u @ np.diag(s)) @ vh
        return F_rank_2

    # convert corresponding points to homogenous coordinates
    ones = np.ones(len(pts1))
    pts1_homogenous = np.hstack((pts1, ones.reshape(-1, 1)))
    pts2_homogenous = np.hstack((pts2, ones.reshape(-1, 1)))
    
    # scaling matrix
    T = np.array(([1/M, 0, 0],
                  [0, 1/M, 0],
                  [0, 0, 1]))
    
    # scale data
    pts1_norm = pts1_homogenous @ T
    pts2_norm = pts2_homogenous @ T
    
    # compute U matrix containing corresponding coordinates
    U = compute_U(pts1_norm, pts2_norm)
    
    # compute normalized Fundamental Matrix (F)
    F_norm = compute_F(U)
    
    # enforce rank 2 constraint (not any F can be a fundamental matrix)
    F_norm = enforce_rank_2_constraint(F_norm)
    
    # unscale normalized fundamental matrix
    F = (T.T @ F_norm) @ T
    
    np.savez('q2_1', F, M)
    return F
'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # derived from: F = K2^(-T) * E * K1^(-1)
    E = K2.T @ F @ K1
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    pass


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
