import numpy as np

def sampleRandomPoints(x1, x2, num_points=4):
	num_rows = x1.shape[0]
	random_i = np.random.choice(num_rows, size=num_points, replace=False)
	rand_coords_x1 = x1[random_i, :]
	rand_coords_x2 = x2[random_i, :]
	return rand_coords_x1, rand_coords_x2

def calculateInliers(x1, x2, H, delta):
	x1_homogenous = np.hstack((x1, np.ones((x1.shape[0], 1))))
	x2_homogenous = np.hstack((x2, np.ones((x2.shape[0], 1))))
	x1_prime = np.array([])
	for i in range(x1_homogenous.shape[0]):
		x1_i_prime = np.matmul(H, np.transpose(x2_homogenous[i, :]))
		x1_i_prime = x1_i_prime / x1_i_prime[-1]
		x1_i_prime = np.transpose(x1_i_prime[:-1])

		if x1_prime.shape[0] == 0:
			x1_prime = np.array(x1_i_prime)
		else:
			x1_prime = np.vstack((x1_prime, x1_i_prime))
	print("x1 prime:", x1_prime)
	dist = np.linalg.norm(x1_prime - x1, axis = 1)
	print("dist:", dist)
	print(delta)
	inliers = np.float32(dist < delta)
	num_inliers = sum(inliers)

	return num_inliers, inliers

def calculateMatches(matches, locs1, locs2):
    x1 = locs1[matches[:, 0]]
    x2 = locs2[matches[:, 1]]
    return np.fliplr(x1), np.fliplr(x2)
