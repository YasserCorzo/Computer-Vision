# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
import os
import skimage
from matplotlib import pyplot as plt
from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    image = np.zeros(res)
    
    # camera intrinsics matrix (no skew, origin of camera and sphere are aligned)
    K = np.array(([pxSize, 0, 0], [0, pxSize, 0], [0, 0, 1]))
    
    for x in range(res[0]):
        for y in range(res[1]):
            # 2d point (added radius since camera is viewing sphere from negative z direction)
            point_2d = np.array([y, x, -rad]).reshape(-1, 1)
            
            # calculate corresponding 3D coordinate of camera pixel coordinate
            point_3d = (K @ point_2d).T.reshape(3,)
            
            # calculate normal vector at 3D point on sphere
            n = (point_3d - center) / np.linalg.norm(point_3d - center)
            
            # calculate observed radiance (pixel value can't be negative)
            I = max(0, np.dot(n, light))
            
            image[x, y] = I    
    
    return image

def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None

    files = os.listdir(path)
    tif_files_path = []
    for file in files:
        if file.endswith('.tif'):
            tif_files_path.append(path + file)
        else:
            L = np.load(path + file).T
            
    print(tif_files_path)
    
    img = np.array(plt.imread(tif_files_path[0]))
    s = img.shape
    P = img.shape[0] * img.shape[1]
    print(P)
    
    I = []
    for file in tif_files_path:
        img = plt.imread(file).astype('uint16')
        i = np.array(img[:, :, 1], dtype='uint16')
        I.append(i.flatten())
    
    I = np.array(I)
    print(I.shape)
    
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = None
    normals = None
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    pass


if __name__ == '__main__':

    # Put your main code here
    
    # Q1.b
    '''
    light_dirs = np.array(([1, 1, 1], [1, -1, 1], [-1, -1, 1])) / np.sqrt(3)
    res = (3840, 2160)
    pxSize = 7 * 10e-6
    rad = 0.75 * 10e-2
    center = np.array([0, 0, 0])
    for i in range(light_dirs.shape[0]):
        image = renderNDotLSphere(center, rad, light_dirs[0, :], pxSize, res)
        plt.figure()
        plt.imshow(image)
        plt.show()
    '''
        
    # Camera parameters
    camera_position = np.array([0, 0, 10])
    pixel_size = 7e-4  # 7 µm in meters
    resolution = (2160, 3840)  # Height x Width

    # Sphere parameters
    sphere_center = np.array([0, 0, 0])
    sphere_radius = 0.0075  # 0.75 cm in meters

    # Lighting directions
    light_directions = [
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, 1]) / np.sqrt(3),
        np.array([-1, -1, 1]) / np.sqrt(3)
    ]
    '''
    # Render images for each lighting direction
    for idx, light_dir in enumerate(light_directions):
        rendered_image = renderNDotLSphere(sphere_center, sphere_radius, light_dir, pixel_size, resolution)

        # Display or save the rendered image
        plt.imshow(rendered_image, cmap='gray')
        plt.title(f"Light Direction {idx+1}")
        plt.show()
    '''   
    # Q1.c
    I, L, s = loadData()
    
    #Q1.d
    U, S, Vh = np.linalg.svd(I, full_matrices=True, compute_uv=True, hermitian=False)
    print("Singular values of I:", np.diagonal(S))
    pass
