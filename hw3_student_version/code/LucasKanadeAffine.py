import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)

    # spline 
    It_spline = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    It1_spline = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    
    # create 3D matrix of homogenous coordinates in template
    rows = np.arange(It.shape[0])
    cols = np.arange(It.shape[1])
    grid_y, grid_x = np.meshgrid(rows, cols)
    grid = np.stack((grid_x, grid_y, np.ones(grid_x.shape)), axis=-1)
    homogenous_coords = grid.reshape(-1, 3).T

    template = It[grid_y, grid_x]
    Iy, Ix = np.gradient(It1)
    Ix_spline = RectBivariateSpline(rows, cols, Ix)
    Iy_spline = RectBivariateSpline(rows, cols, Iy)
    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    for i in range(int(num_iters)):
        # get warped coordinates 
        transform_matrix = np.array(([1.0 + p[0], p[1], p[2]], 
                                     [p[3], 1.0 + p[4], p[5]], 
                                     [0, 0, 1.0]))
        warped_coords = transform_matrix @ homogenous_coords
        
        # put warped coordinates in a grid (cols x rows x 3)
        warped_grid = warped_coords.T.reshape(It.shape[1], It.shape[0], 3)
        
        # convert homogenous coordinates in grid to heterogenous
        warped_grid[:, :, 0] /= warped_grid[:, :, 2]
        warped_grid[:, :, 1] /= warped_grid[:, :, 2]

        # retrieve coordinates in warped image that don't image boundaries
        x_thresh = len(cols)
        y_thresh = len(rows)
        valid_warp_grid = (warped_grid[:, :, 0] < x_thresh) & (warped_grid[:, :, 0] >= 0) & (warped_grid[:, :, 1] >= 0) & (warped_grid[:, :, 1] <= y_thresh)

        # get pixel values at warped image
        I_w = It1_spline.ev((warped_grid[:, :, 1]), (warped_grid[:, :, 0]))
        '''
        # get matching warped coordinates in template
        homogenous_warped_coords = np.vstack((warped_coords, np.ones(warped_coords.shape[1])))
        template_coords = np.linalg.inv(transform_matrix) @ homogenous_warped_coords

        # retrieve values of matching coords in template
        T = it_spline.ev(template_coords[1, :], template_coords[0, :])
        '''
        # compute error image (cols x rows)
        error_img = template - I_w
        
        # locations where coords are not valid set them to 0
        error_img[valid_warp_grid == False] = 0
        
        # reshape error
        error_img = error_img.reshape(-1, 1)
        
        # compute image gradient of warped image
        I_y = Iy_spline.ev(warped_grid[:, :, 1], warped_grid[:, :, 0])
        I_x = Ix_spline.ev(warped_grid[:, :, 1], warped_grid[:, :, 0])
        image_gradient = np.vstack((I_x.ravel(), I_y.ravel())).T
        
        # compute dW_dp for all points in the rectangle
        j_image_gradient = []
        
        for j in range(warped_coords.shape[1]):
            x = warped_coords[0, j]
            y = warped_coords[1, j]
            dW_dp_i = np.array(([x, y, 1, 0, 0, 0], 
                                [0, 0, 0, x, y, 1]))
            img_grad_i = image_gradient[j, :]
            j_image_gradient.append(img_grad_i @ dW_dp_i)
        
        # m x 6 
        j_image_gradient = np.array(j_image_gradient)
        
        # 6 x 6
        H = j_image_gradient.T @ j_image_gradient
        delta_p = np.linalg.inv(H) @ j_image_gradient.T @ error_img
        delta_p = delta_p.T[0]
        if np.linalg.norm(delta_p) <= threshold:
            break
        
        # update p
        p += delta_p
    
    M = np.array(([1.0+p[0], p[1], p[2]], [p[3],1.0+p[4], p[5]], [0, 0, 1.0]))
    return M