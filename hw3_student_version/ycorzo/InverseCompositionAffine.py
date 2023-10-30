import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
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

    template = It_spline.ev(grid[:, :, 1], grid[:, :, 0])
    
    # calculate image gradient of template 
    Iy, Ix = np.gradient(It)
    temp_gradient = np.vstack((Ix.ravel(), Iy.ravel())).T
    
    # calculate template gradient * Jacobian 
    j_template_gradient = []
        
    for j in range(homogenous_coords.shape[1]):
        x = homogenous_coords[0, j]
        y = homogenous_coords[1, j]
        dW_dp_i = np.array(([x, y, 1, 0, 0, 0], 
                            [0, 0, 0, x, y, 1]))
        temp_grad_i = temp_gradient[j, :]
        j_template_gradient.append(temp_grad_i @ dW_dp_i)
    
    j_template_gradient = np.array(j_template_gradient)
    
    # calculate Hessian
    H = j_template_gradient.T @ j_template_gradient
    
    for i in range(int(num_iters)):
        # get warped coordinates 
        warped_coords = M @ homogenous_coords
        
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
    
        # compute error image (cols x rows)
        error_img = template - I_w
        
        # locations where coords are not valid set them to 0
        error_img[valid_warp_grid == False] = 0
        
        # reshape error
        error_img = error_img.reshape(-1, 1)
        
        delta_p = np.linalg.inv(H) @ j_template_gradient.T @ error_img
        delta_p = delta_p.T[0]
        
        if np.linalg.norm(delta_p) <= threshold:
            break
        
        # update p
        p += delta_p
        
        # update M
        delta_M = np.vstack((delta_p.reshape(2,3), [0, 0, 1]))
        M = M @ np.linalg.inv(delta_M)
    
    M = np.array(([1.0+p[0], p[1], p[2]], [p[3],1.0+p[4], p[5]], [0, 0, 1.0]))
    return M
