import numpy as np
import cv2

from myHelperFunctions import *

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points

	# first compute A (matrix derived from points in x1 and x2)

	# N is number of corresponding points in x1 and x2
	N = x1.shape[0]
	A = np.zeros((N * 2, 9))

	# iterate through x2, retrieve corresponding points in x1 to fill up A
	for i in range(N):
		(x, y) = (x2[i, 0], x2[i, 1])
		(u, v) = (x1[i, 0], x1[i, 1])
		A[i * 2 : (i * 2) + 2] = np.array(([x, y, 1, 0, 0, 0, -x * u, -y * u, -u], [0, 0, 0, x, y, 1, -x * v, -y * v, -v]))

	# compute small h that solves Ah = 0
	u, s, v = np.linalg.svd(A, full_matrices=True, compute_uv=True, hermitian=False)
	# h equals the eigenvector corresponding to the zero eigenvalue. Thus, we choose the smallest eigenvalue 
	# of A^T*A, which is σ9 in s and the least-squares solution to Ah = 0 is the corresponding eigenvector 
	# (in column 9 of the matrix V).
	h = v[-1]

	# h is a 9x1 vector containing elements in H (3x3). Reshape h to a 3x3 matrix
	H2to1 = h.reshape(3, 3)

	return H2to1



def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	x1_centroid = np.mean(x1, axis=0)
	x2_centroid = np.mean(x2, axis=0)

	#Shift the origin of the points to the centroid
	x1_shifted = x1 - x1_centroid
	x2_shifted = x2 - x2_centroid
	
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	max_dist_x1 = np.amax(np.linalg.norm(x1_shifted, axis=1))
	scale1 = np.sqrt(2) / max_dist_x1

	max_dist_x2 = np.amax(np.linalg.norm(x2_shifted, axis=1))
	scale2 = np.sqrt(2) / max_dist_x2

	x1_norm = scale1 * x1_shifted
	x2_norm = scale2 * x2_shifted

	#Similarity transform 1
	T1 = np.array(([scale1, 0, -scale1 * x1_centroid[0]], [0, scale1, -scale1 * x1_centroid[1]], [0, 0, 1]))

	#Similarity transform 2
	T2 = np.array(([scale2, 0, -scale2 * x2_centroid[0]], [0, scale2, -scale2 * x2_centroid[1]], [0, 0, 1]))

	#Compute homography
	H_norm = computeH(x1_norm, x2_norm)

	#Denormalization
	H2to1 = (np.linalg.inv(T1) @ H_norm) @ T2

	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	max_num_inliers = 0
	inliers = np.zeros(locs1.shape[0])

	for i in range(max_iters):
		# randomly sample at least 4 points
		x1_sample, x2_sample = sampleRandomPoints(locs1, locs2, num_points=4)

		# compute H using sampled points
		H_norm = computeH_norm(x1_sample, x2_sample)

		# calculate number of inliers as well as inliers using H computed from sample correspondences
		ith_num_inliers, ith_inliers_vec = calculateInliers(locs1, locs2, H_norm, inlier_tol)
		
		# update number of inliers and inliers vector
		if ith_num_inliers > max_num_inliers:
			max_num_inliers = ith_num_inliers
			inliers = ith_inliers_vec

	# get which points in inliers contains a 1.
	# those will be the ith corresponding points in locs1 and locs2 that compute the best fitting H
	consensus_set_indexes = np.where(inliers == 1)[0]
	bestH2to1 = computeH_norm(locs1[consensus_set_indexes], locs2[consensus_set_indexes])

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template
	white = np.zeros((template.shape[0], template.shape[1], 3))
	white.fill(255)
	
	#Warp mask by appropriate homography
	warp_mask = cv2.warpPerspective(white, H2to1, (img.shape[1], img.shape[0]))
	
	#Warp template by appropriate homography
	warp_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))

	#Use mask to combine the warped template and the image
	not_warp_mask = np.logical_not(warp_mask)
	apply_mask = np.multiply(img, not_warp_mask)
	
	composite_img = apply_mask + warp_template
	
	return composite_img 


