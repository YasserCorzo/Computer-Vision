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
    It1_spline = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    # get all possible homogenous coordinate positions of rectangle in template
    rows = np.arange(It.shape[0])
    cols = np.arange(It.shape[1])
    x_coords = np.tile(rows, len(cols))
    y_coords = np.repeat(cols, len(rows))
    homogenous_rect_coords = np.array([x_coords, y_coords, np.ones(rows.shape[0] * cols.shape[0])])

    for i in range(int(num_iters)):
        # get warped coordinates 
        transform_matrix = np.array(([1.0 + p[0], p[1], p[2]], 
                                     [p[3], 1.0 + p[4], p[5]], 
                                     [0, 0, 1.0]))
        homogenous_warped_coords = transform_matrix @ homogenous_rect_coords
        warped_coords = homogenous_warped_coords[:-1]

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
        I_w = It1_spline.ev(warped_coords[1, :], warped_coords[0, :])

        # get matching warped coordinates in template
        homogenous_warped_coords = np.vstack((warped_coords, np.ones(warped_coords.shape[1])))
        template_coords = np.linalg.inv(transform_matrix) @ homogenous_warped_coords

        # retrieve values of matching coords in template
        T = it_spline.ev(template_coords[1, :], template_coords[0, :])

        # compute error image (m x 1)
        error_img = T - I_w

        # compute image gradient of warped image
        I_y = np.array([It1_spline.ev(warped_coords[1, :], warped_coords[0, :], dx=1)]).T
        I_x = np.array([It1_spline.ev(warped_coords[1, :], warped_coords[0, :], dy=1)]).T
        image_gradient = np.hstack((I_x, I_y))
    
        # m x 2
        image_gradient = np.hstack((I_x, I_y))

        # compute dW_dp for all points in the rectangle
        j_image_gradient = []
    
        for j in range(warped_coords.shape[1]):
            x = warped_coords[0, j]
            y = warped_coords[1, j]
            dW_dp_i = np.array(([x, 0, y, 0, 1, 0], 
                                [0, x, 0, y, 0, 1]))
            img_grad_i = image_gradient[j, :]
            j_image_gradient.append(img_grad_i @ dW_dp_i)
        # m x 6 
        j_image_gradient = np.array(j_image_gradient)

        # 6 x 6
        H = j_image_gradient.T @ j_image_gradient
        delta_p = np.linalg.inv(H) @ j_image_gradient.T @ error_img
        
        if np.linalg.norm(delta_p) <= threshold:
           break
           
        # update p
        p += delta_p
    
    M = np.array(([1.0+p[0], p[1], p[2]], [p[3],1.0+p[4], p[5]], [0, 0, 1.0]))
    return M
