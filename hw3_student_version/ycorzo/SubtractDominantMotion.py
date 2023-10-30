import numpy as np

from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
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
    
    #M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    
    # spline 
    image1_spline = RectBivariateSpline(np.arange(image1.shape[0]), np.arange(image1.shape[1]), image1)
    
    # create 3D matrix of homogenous coordinates in image1
    rows = np.arange(image1.shape[0])
    cols = np.arange(image1.shape[1])
    grid_y, grid_x = np.meshgrid(rows, cols)
    grid = np.stack((grid_x, grid_y, np.ones(grid_x.shape)), axis=-1)
    homogenous_coords = grid.reshape(-1, 3).T

    # calculate warped coordinates
    warped_coords = np.linalg.inv(M) @ homogenous_coords
    
    # put warped coordinates in a grid (cols x rows x 3)
    warped_grid = warped_coords.T.reshape(image1.shape[1], image1.shape[0], 3)

    # convert homogenous coordinates in grid to heterogenous
    warped_grid[:, :, 0] = warped_grid[:, :, 0] / warped_grid[:, :, 2]
    warped_grid[:, :, 1] = warped_grid[:, :, 1] / warped_grid[:, :, 2]

    # retrieve coordinates in warped image that don't image boundaries
    x_thresh = len(cols)
    y_thresh = len(rows)
    valid_warp_grid = (warped_grid[:, :, 0] < x_thresh) & (warped_grid[:, :, 0] >= 0) & (warped_grid[:, :, 1] >= 0) & (warped_grid[:, :, 1] < y_thresh)

    # retrieve valid coordinate values in image1
    image1_patch = image1_spline.ev(warped_grid[:, :, 1], warped_grid[:, :, 0])

    # retrieve valid coordinate values in image2
    image2_patch = image2[grid_y, grid_x]

    # find absolute difference
    diff = np.abs(image2_patch - image1_patch)

    # set locations where valid coordinates are to 0
    diff[valid_warp_grid == False] = 0
    
    # set locations where difference exceeds tolerance to 1
    mask[grid_y, grid_x] = diff > tolerance

    # erode mask (shrink dark regions and enlarge light regions)
    #mask = binary_erosion(mask)

    # dilate mask (enlarge dark regions and shrink light regions)
    mask = binary_dilation(mask, np.ones((5, 5)))
    mask = binary_erosion(mask, np.ones((5, 5)))

    return mask
