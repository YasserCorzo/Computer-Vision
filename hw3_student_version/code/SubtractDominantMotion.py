import numpy as np

from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage import affine_transform
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.interpolate import RectBivariateSpline

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    # calculate M (affine transformation matrix)
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)

    # spline image1
    image1_spline = RectBivariateSpline(np.arange(image1.shape[0]), np.arange(image1.shape[1]), image1)

    # create 3D matrix of homogenous coordinates in image1
    rows = np.arange(image1.shape[0])
    cols = np.arange(image1.shape[1])
    grid_y, grid_x = np.meshgrid(cols, rows)
    grid = np.stack((grid_x, grid_y, np.ones(grid_x.shape)), axis=-1)
    homogenous_coords = grid.reshape(-1, 3).T

    # calculate warped coordinates
    warped_coords = M @ homogenous_coords
    
    # put warped coordinates in a grid (rows x cols x 2)
    warped_grid = warped_coords[:2, :].T.reshape(len(rows), len(cols), 2)

    # retrieve warped pixel values
    image1_w = image1_spline.ev(warped_grid[:, :, 1], warped_grid[:, :, 0])

    # retrieve coordinates in warped image that exceed image boundaries
    x_thresh = len(rows)
    y_thresh = len(cols)
    i = np.where((warped_coords[0] >= x_thresh) | (warped_coords[1] >= y_thresh) | (warped_coords[0] < 0) | (warped_coords[1] < 0))[0]
    not_valid_coords = warped_coords[:, i]

    # find matching not valid coords in image1
    not_valid_coords_img1 = (np.linalg.inv(M) @ not_valid_coords).T[:, :-1].astype('int')

    # set not valid coords in warped image1 to 0
    #image1_w[not_valid_coords_img1[:, 0], not_valid_coords_img1[:, 1]] = 0
   
    # calculate absolute difference between warped img and image at time t+1
    abs_diff = np.abs(image2 - image1_w)
    #abs_diff[not_valid_coords_img1[:, 1], not_valid_coords_img1[:, 0]] = 0

    # calculate locations where difference exceeds threshold
    mask = abs_diff > tolerance

    # erode mask (shrink bright regions and enlarge dark regions)
    structuring_element = np.ones((3, 3))
    mask = binary_erosion(mask, structuring_element)

    # dilate mask (enlarge bright regions and shrink dark regions)
    mask = binary_dilation(mask, structuring_element)

    return mask.astype(int)
