"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import trace
import cv2
import numpy as np

from util import _singularize
from scipy.ndimage import gaussian_filter

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
    F_norm = _singularize(F_norm)
    
    # unscale normalized fundamental matrix
    F = (T.T @ F_norm) @ T
    
    #np.savez('q2_1', F, M)
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
    w = []
    err = []
    w_homogenous = []

    # to build A: x1_i x P1 & x2_i x P2
    # remember: a x b = [[a2b3 - a3b2], [a3b1 - a1b3], a1b2 - a2b1]
    # in this case, x x C will yield 3 equations, one of which is redundant (a linear combination of the other two), and 4 variables
    # the variables: y*c3, c2, c1, x*c3 (ci is row i of camera matrix)
    # source: https://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf

    # get rows of camera matrix 1
    c1 = C1[0, :]
    c2 = C1[1, :]
    c3 = C1[2, :]

    # get rows of camera matrix 2
    c1_2 = C2[0, :]
    c2_2 = C2[1, :]
    c3_2 = C2[2, :]

    # solve for w
    for i in range(len(pts1)):
        x1_i, y1_i = pts1[i, :]
        x2_i, y2_i = pts2[i, :]
    
        A1_i = np.array((y1_i*c3 - c2, c1 - x1_i*c3))
        A2_i = np.array((y2_i*c3_2 - c2_2, c1_2 - x2_i*c3_2))
        
        A_i = np.vstack((A1_i, A2_i))
      
        # use SVD to solve for A_i * w_i = 0
        u, s, vh = np.linalg.svd(A_i, full_matrices=True, compute_uv=True, hermitian=False)
        w_i_homogenous = vh[-1]
        
        # convert homogenous w_i to heterogenous w_i (x, y, z)
        w_i = w_i_homogenous[:-1] / w_i_homogenous[-1]
        w.append(w_i)
        w_homogenous.append(w_i_homogenous)
        
    w = np.array(w)
    w_homogenous = np.array(w_homogenous)

    proj1 = (C1 @ w_homogenous.T).T
    proj2 = (C2 @ w_homogenous.T).T

    pts1_hat = proj1[:, :-1] / proj1[:, -1].reshape(-1, 1)
    pts2_hat = proj2[:, :-1] / proj2[:, -1].reshape(-1, 1)

    err1 = np.square(np.linalg.norm(pts1 - pts1_hat, axis=1))
    err2 = np.square(np.linalg.norm(pts2 - pts2_hat, axis=1))
    
    # calculate total reprojection error
    err = (err1 + err2).sum()

    return w, err

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
    def window_center_coord(size):
        return (int(size / 2), int(size / 2))

    def epipolar_line_pts(im2, point, F):
        candidate_points = []
        l_prime = F @ point.reshape(-1, 1)
        a = l_prime[0, 0]
        b = l_prime[1, 0]
        c = l_prime[2, 0]
        for x in range(im2.shape[1]):
            y = (-a*x - c)/b 
            y = round(y)
            if y < im2.shape[0] and y >=0:
                candidate_points.append([x, y])
        
        candidate_points = np.array(candidate_points)

        return candidate_points
    
    def window(y, x, img, center):
        cy, cx = center
        windowed_img = img[y-cy : y+cy+1, x-cx : x+cx+1]
        return windowed_img

    window_size = 225
    window_center = window_center_coord(window_size)
    paddingX = window_size // 2
    paddingY = window_size // 2
    
    # convert to greyscale to pad
    if (im1.ndim == 3):
        im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)

    if (im2.ndim == 3):
        im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    
    # normalize grey images
    im1 = np.float32(im1) / 255 
    im2 = np.float32(im2) / 255
    
    im1 = gaussian_filter(im1, sigma=1)
    im2 = gaussian_filter(im2, sigma=1)
    
    paddedImg1 = np.pad(im1, pad_width=((paddingX,),(paddingY,)), mode='edge')
    paddedImg2 = np.pad(im2, pad_width=((paddingX,),(paddingY,)), mode='edge')

    pt1 = np.array([x1, y1, 1])
    epipolar_pts = epipolar_line_pts(im2, pt1, F)

    # obtain window in image 1 (padded)
    padded_y1_i = y1 + paddingY
    padded_x1_i = x1 + paddingX
    window1 = window(padded_y1_i, padded_x1_i, paddedImg1, window_center)
    
    # score for euclidean distance
    euclidean_score = float('inf')
    
    # keep track of (x, y) coordinate that has best window similarity
    best_x2 = None
    best_y2 = None
    
    # obtain windows for points on epipolar line in image 2
    for i in range(len(epipolar_pts)):
        x2_i, y2_i = epipolar_pts[i, :]
        
        # obtain window in image 2 (padded)
        padded_y2_i = y2_i + paddingY
        padded_x2_i = x2_i + paddingX
        window2 = window(padded_y2_i, padded_x2_i, paddedImg2, window_center)
        
        # choose corresponding point in image 2 whose window has with the smallest euclidean distance
        score = np.sqrt(np.sum(np.square(window1 - window2)))
        if score < euclidean_score:
            euclidean_score = score
            best_x2 = x2_i
            best_y2 = y2_i
        
    x2, y2 = best_x2, best_y2
    return x2, y2

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
    inliers = None
    max_num_inliers = float('-inf')

    # make points homogenous
    pts1_homogenous = np.hstack((pts1, np.ones(len(pts1)).reshape(-1, 1)))
    pts2_homogenous = np.hstack((pts2, np.ones(len(pts2)).reshape(-1, 1)))

    for iter in range(nIters):
        # draw set with minimum number of correspondences (7 points)
        i = np.random.randint(0, len(pts1), 7)
        pts1_rand = pts1[i]
        pts2_rand = pts2[i]

        # fit F to set
        curr_F = eightpoint(pts1_rand, pts2_rand, M)
    
        # retrieve fitted epipolar line
        l_prime = (curr_F @ pts1_homogenous.T).T

        # calculate distances between every pt2 and the fitted epipolar line
        dist = []
        for i in range(len(pts1)):
            pts2_i = pts2_homogenous[i, :]
            l_i = l_prime[i, :]

            # use geometric distance
            dist_i = np.dot(pts2_i, l_i) / np.sqrt(l_i[0]**2 + l_i[1]**2)
            dist.append(dist_i)
        dist = np.array(dist)
        
        inliers = dist < tol
        curr_num_inliers = inliers.sum()

        if curr_num_inliers > max_num_inliers:
            max_num_inliers = curr_num_inliers
            inliers = np.argwhere(dist)

    F = eightpoint(pts1[inliers.flatten()], pts2[inliers.flatten()], M)
    return F, inliers
    
'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    I = np.eye(3)
    if theta == 0:
        return I
    r = r / theta
    kx = r[0, 0]
    ky = r[1, 0]
    kz = r[2, 0]
    K = np.array(([0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]))
    return I*np.cos(theta) + (1 - np.cos(theta))*(r @ r.T) + K*np.sin(theta)
    
'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    '''
    theta = np.arccos((np.trace(R) - 1) / 2)
    w = (1 / (2 * np.sin(theta))) * np.array(([R[2, 1] - R[1, 2]],
                                              [R[0, 2] - R[2, 0]], 
                                              [R[1, 0] - R[0, 2]]))  
    r = theta * w
    '''
    # source: https://courses.cs.duke.edu/cps274/fall13/notes/rodrigues.pdf
    A = (R - R.T) / 2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]]).reshape(-1, 1)
    s = np.linalg.norm(rho)
    c = (np.trace(R) - 1) / 2
    if (s == 0) and (c == 1):
        r = np.zeros((3, 1))
    
    elif (s == 0) and (c == -1):
        # let v be a non-zero column of R + I
        v_temp = R + np.eye(len(R))
        v = None
        for j in range(v_temp.shape[1]):
            # check whether column j is non-zero
            col = v_temp[:, j]
            if np.all(col != 0):
                v = col
                break
        u = v / np.linalg.norm(v)
        r = u * np.pi
        # modeled after function S_1/2
        if (np.linalg.norm(r) == np.pi) and ((r[0] == 0 and r[1] == 0 and r[2] < 0) 
            or (r[0] == 0 and r[1] < 0) or (r[0] < 0)):
            r = -r
        r = r.reshape((3, 1))
        
    theta = np.arccos((np.trace(R) - 1) / 2)
    if (np.sin(theta) != 0):
        u = rho / s
        theta = np.arctan2(s, c)
        r = u * theta 
    return r

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
    # retrieve number of 2D coordinates
    N = p1.shape[0]

    # break up x into 3D coords P, r2, and t2
    t2 = x[-3:].reshape(3, 1)
    r2 = x[-6:-3].reshape(3, 1)
    P = x[:3 * N].reshape(N, 3)
    
    # make 3D coordinate homogenous (N x 4)
    P_homogenous = np.hstack((P, np.ones(N).reshape(-1, 1)))
    
    # after getting r2, get R2 (with Rodriguez function)
    R2 = rodrigues(r2)
    
    # after getting R2, get M2 = [R2 | t2]
    M2 = np.hstack((R2, t2))

    # get C1 (K1 @ M1) and C2 (K2 @ M2)
    C1 = K1 @ M1
    C2 = K2 @ M2
    
    # calculate p1_hat and p2_hat (multiply Ci @ 3D coords) 
    p1_hat_homogenous = (C1 @ P_homogenous.T).T
    p2_hat_homogenous = (C2 @ P_homogenous.T).T

    # convert homogenous coordinates to heterogenous
    p1_hat = p1_hat_homogenous[:, :-1] / p1_hat_homogenous[:,-1].reshape(-1, 1)
    p2_hat = p2_hat_homogenous[:, :-1] / p2_hat_homogenous[:, -1].reshape(-1, 1)

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    return residuals

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
