import numpy as np

def sampleRandomPoints(x1, x2, num_points=4):
	num_rows = x1.shape[0]
	# choose randomly 4 rows from x1 and x2, equivalent to choosing 4 random matched points
	random_i = np.random.choice(num_rows, size=num_points, replace=False)
	rand_coords_x1 = x1[random_i, :]
	rand_coords_x2 = x2[random_i, :]
	return rand_coords_x1, rand_coords_x2

def calculateInliers(x1, x2, H, delta):
	x2_homogenous = np.hstack((x2, np.ones((x2.shape[0], 1))))
	x1_homogenous_prime = (H @ x2_homogenous.T).T
	x1_prime = x1_homogenous_prime[:, :-1] / x1_homogenous_prime[:, [-1]]
	dist = np.linalg.norm(x1_prime - x1, axis = 1)
	inliers = np.float32(dist < delta)
	num_inliers = sum(inliers)

	return num_inliers, inliers

def findMatches(matches, locs1, locs2):
	# invert coordinates because coordinate system in cv2 is different from numpy
	# i.e. (y, x) = (row, col) --> (x, y) = (row, col)
    x1 = np.fliplr(locs1[matches[:, 0]])
    x2 = np.fliplr(locs2[matches[:, 1]])
    return x1, x2

def cropToAspectRatio(img_to_crop, img_reference):
	aspect_ratio = img_reference.shape[1] / img_reference.shape[0]
	crop_width = img_to_crop.shape[0] * aspect_ratio

	# find center of image that needs to be cropped (remember in cv2 (x, y) = (cols, rows))
	cx, cy = img_to_crop.shape[1] // 2, img_to_crop.shape[0] // 2

	# crop image from center with required width
	dx = int(crop_width // 2)
	crop_img = img_to_crop[:, (cx - dx) : (cx + dx + 1), :]
	return crop_img