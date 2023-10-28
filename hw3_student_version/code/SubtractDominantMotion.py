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
    '''
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    
    # spline 
    image1_spline = RectBivariateSpline(np.arange(image1.shape[0]), np.arange(image1.shape[1]), image1)
    image2_spline = RectBivariateSpline(np.arange(image2.shape[0]), np.arange(image2.shape[1]), image2)
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
    image1_w = image2_spline.ev(warped_grid[:, :, 1], warped_grid[:, :, 0])

    # retrieve coordinates in warped image that exceed image boundaries
    x_thresh = len(rows)
    y_thresh = len(cols)
    i = np.where((warped_coords[0] <= x_thresh) & (warped_coords[0] >= 0) & (warped_coords[1] >= 0) | (warped_coords[1] <= y_thresh))[0]
    valid_coords = warped_coords[:, i]

    # find matching not valid coords in image1
    valid_coords_img1 = (np.linalg.inv(M) @ valid_coords).T[:, :-1]

    # grid of valid coords
    valid_coords_grid = valid_coords_img1.T.reshape(image1.shape[0], image1.shape[1], 2)

    # retrieve valid coordinate values in image1
    image1_valid = image1_spline.ev(valid_coords_grid[:, :, 1], valid_coords_grid[:, :, 0])

    # warp new image 1
    image1_w = affine_transform(image1_valid, np.linalg.norm(M))

    # find absolute difference
    diff = np.abs(image2 - image1_w)

    # mask the difference (place 1 where diff > tolerance)
    mask = diff > tolerance

    structuring_element = np.ones((3, 3))

    # erode mask (shrink dark regions and enlarge light regions)
    mask = binary_erosion(mask)

    # dilate mask (enlarge dark regions and shrink light regions)
    mask = binary_dilation(mask)

    return mask
    '''

    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = M[:2,:]
    M = np.concatenate([M, np.array([0,0,1])[np.newaxis,:]])

    # Eliminate the invalid coordinates
    It_interp = RectBivariateSpline(range(image1.shape[0]), range(image1.shape[1]), image1)
    h,w = image1.shape
    x_linspace = np.arange(w)
    y_linspace = np.arange(h)
    y, x = np.meshgrid(y_linspace, x_linspace)

    original_coordinate = np.stack([x, y, np.ones_like(x)], axis=-1)
    # print("M {} original_coordinate {} y {}".format(M.shape, original_coordinate.shape, y.shape)) # (320, 240, 3, 3)
    tiled_M = np.tile(np.linalg.inv(M), (y.shape[0], y.shape[1], 1, 1)) # tiled transformation matrix for each point
    # print("tiled_M", tiled_M.shape) # (320, 240, 3, 3)
    warpped_coordinates = tiled_M @ original_coordinate[:,:,:,np.newaxis]
    warpped_coordinates = np.squeeze(warpped_coordinates)
    # convert to x-y coordinate
    warpped_coordinates[:,:,0] /= warpped_coordinates[:,:,2]
    warpped_coordinates[:,:,1] /= warpped_coordinates[:,:,2]
    x_warp = warpped_coordinates[:,:,0]
    y_warp = warpped_coordinates[:,:,1]
    valid_warp = (warpped_coordinates[:,:,0]>=0) & (warpped_coordinates[:,:,0]<image1.shape[1]) & (warpped_coordinates[:,:,1]>=0)& (warpped_coordinates[:,:,1]<image1.shape[0])

    It_patch = It_interp.ev(y_warp,x_warp)
    It1_patch = image2[y, x]
    err_img = np.abs(It1_patch - It_patch)
    err_img[valid_warp==False] = 0

    mask[y, x] = (err_img > tolerance)

    # mask = binary_erosion(mask)
    mask = binary_dilation(mask)

    return mask