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

    # spline template and image
    it_spline = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    transform_spline = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    # create coords matrix of current image
    rows = np.arange(It1.shape[0])
    cols = np.arange(It1.shape[1])
    x_coords = np.tile(rows, len(cols))
    y_coords = np.repeat(cols, len(rows))
    homogenous_coords = np.array([x_coords, y_coords, np.ones(rows.shape[0] * cols.shape[0])])

    for iter in range(int(num_iters)):
        # calculate warped coordinates
        warped_coords = M @ homogenous_coords

        # remove warped coordinates outside of range of template
        x_threshold, y_threshold = It.shape[0] - 1, It.shape[1] - 1
        indices_x_threshold = np.where(warped_coords[0] > x_threshold)[0]
        warped_coords = np.delete(warped_coords, indices_x_threshold, axis=1)
        i = np.where(warped_coords[0] < 0)[0]
        warped_coords = np.delete(warped_coords, i, axis=1)
        indices_y_threshold = np.where(warped_coords[1] > y_threshold)[0]
        warped_coords = np.delete(warped_coords, indices_y_threshold, axis=1)
        i = np.where(warped_coords[1] < 0)[0]
        warped_coords = np.delete(warped_coords, i, axis=1)

        # get pixel values at warped image
        I_w = transform_spline.ev(warped_coords[1, :], warped_coords[0, :])

        # get matching warped coordinates in template
        homogenous_warped_coords = np.vstack((warped_coords, np.ones(warped_coords.shape[1])))
        template_coords = np.linalg.inv(np.vstack((M, np.array([0, 0, 1])))) @ homogenous_warped_coords

        # retrieve values of matching coords in template
        T = it_spline.ev(template_coords[1, :], template_coords[0, :])

        # compute error image
        error = T - I_w

        # compute Image gradient (of warped image)
        I_dx = np.array([transform_spline.ev(warped_coords[1, :], warped_coords[0, :], dy=1)]).T
        I_dy = np.array([transform_spline.ev(warped_coords[1, :], warped_coords[0, :], dx=1)]).T
        image_gradient = np.hstack((I_dx, I_dy))

        # compute dW_dp for all points in the rectangle
        dW_dp_all = []
    
        for j in range(warped_coords.shape[1]):
            x = warped_coords[0, j]
            y = warped_coords[1, j]
            dW_dp_i = np.array(([x, 0, y, 0, 1, 0], 
                                [0, x, 0, y, 0, 1]))
            dW_dp_all.append(dW_dp_i)
        # m x 2 x 6 
        dW_dp = np.array(dW_dp_all)
        
        # compute J for all points in warped coordinates
        J_all = []

        for m in range(warped_coords.shape[1]):
            J_i = image_gradient[m, :] @ dW_dp[m, :, :]
            J_all.append(J_i)

        # m x 6
        J = np.array(J_all)
        
        #J = np.einsum('ij,ijk->ik', image_gradient, dW_dp)
        # compute delta p
        delta_p = np.linalg.lstsq(J, error, rcond=-1)[0]
        if np.linalg.norm(delta_p) <= threshold:
           break
        
        # update p
        p += delta_p

        # update M
        M = np.array(([1.0 + p[0], p[1], p[2]], 
                      [p[3], 1.0 + p[4], p[5]]))
    
    M = np.vstack((M, np.array([0, 0, 1])))
    return M
