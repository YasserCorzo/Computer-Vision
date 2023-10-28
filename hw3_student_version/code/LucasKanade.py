import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0

    # retrieve top left and bottom right coordinates
    x1, y1, x2, y2 = rect

    # interpolate template and current image
    temp_interpolate = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    img_interpolate = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    
    # get all possible homogenous coordinate positions of rectangle in template
    rows = np.arange(x1, x2 + 1)
    cols = np.arange(y1, y2 + 1)
    x_coords = np.tile(rows, len(cols))
    y_coords = np.repeat(cols, len(rows))
    homogenous_rect_coords = np.array([x_coords, y_coords, np.ones(rows.shape[0] * cols.shape[0])])

    # compute gradient of every coordinate in rectangle
    template = temp_interpolate.ev(y_coords, x_coords)

    for i in range(int(num_iters)):
        # get warped coordinates (under simple translation)
        translation_matrix = np.array(([1, 0, p[0]], [0, 1, p[1]], [0, 0, 1]))
        homogenous_warped_coords = translation_matrix @ homogenous_rect_coords
        warped_coords = homogenous_warped_coords[:-1]

        # get pixel values in warped image
        I_w = img_interpolate.ev(warped_coords[1, :], warped_coords[0, :])

        # compute error image
        error_img = template - I_w

        # compute image gradient of warped image
        I_y = np.array([img_interpolate.ev(warped_coords[1, :], warped_coords[0, :], dx=1)]).T
        I_x = np.array([img_interpolate.ev(warped_coords[1, :], warped_coords[0, :], dy=1)]).T
        image_gradient = np.hstack((I_x, I_y))
        
        # compute jacobian * image gradient
        J = np.eye(2) # only two warp parameters: p1, p2 --> W(x ; p) : [[x + p1], [y + p2]]
        J_gradient_img = image_gradient @ J

        H = J_gradient_img.T @ J_gradient_img
        '''
        # compute delta p
        delta_p = np.linalg.lstsq(J_gradient_img, error_img, rcond=-1)[0]
        '''
        delta_p = np.linalg.inv(H) @ J_gradient_img.T @ error_img
        if np.linalg.norm(delta_p) <= threshold:
            break
        
        # update p
        p += delta_p
    
    return p