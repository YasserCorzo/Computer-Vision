# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import cv2
import numpy as np
import os
import scipy
import skimage
from matplotlib import pyplot as plt
from matplotlib import cm
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
    
    for i in range(res[0]):
        for j in range(res[1]):
            # subtract by half of camera dims to ensure that the coordinate system is centered at the middle of the frame
            point = np.array([(i - res[0]/2) * pxSize, (j- res[1]/2) * pxSize])
            x, y = point[0], point[1]

            # center coords of sphere
            (a, b, c) = center

            # solve for z
            z = np.lib.scimath.sqrt(np.square(rad) - np.square(x - a) - np.square(y - b)) + c

            point = np.array([x - a, y - b, z - c])
            
            # calculate normal vector at 3D point on sphere
            n = point / np.linalg.norm(point)
            
            # calculate observed radiance (pixel value can't be negative)
            I = max(0, np.dot(n, light))
            
            image[i, j] = I    
    
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
    
    img = np.array(plt.imread(tif_files_path[0]))
    s = (img.shape[0], img.shape[1])
    
    I = []
    for file in tif_files_path:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        i = np.array(img[:, :, 1])
        I.append(i.flatten())
    
    I = np.array(I)
    
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

    B, _, _, _ = np.linalg.lstsq(L @ L.T, L @ I)
    print(B)

    print(np.linalg.inv((L @ L.T)) @ L @ I)

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

    # dimension (P,)
    albedos = np.linalg.norm(B, axis=0)
    print(albedos.shape)

    # 3 x P
    normals = (B.T / albedos.reshape(-1, 1)).T
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

    albedoIm = albedos.reshape(s)

    normalIm = [normals[i].reshape(s) for i in range(normals.shape[0])]
    normalIm = np.stack(normalIm, axis=2)

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

    # horizontal and vertical gradients are every pixel p
    zx = np.zeros(normals.shape[1])
    zy = np.zeros(normals.shape[1])

    # calculate the x and y derivative at every pixel p
    for p in range(normals.shape[1]):
        pixel_norm = normals[:, p] 
        fx = -pixel_norm[0] / pixel_norm[-1]
        fy = -pixel_norm[1] / pixel_norm[-1]
        zx[p] = fx
        zy[p] = fy
    
    # add 3 channels to gradients (dim s)
    #zx = [zx.reshape((s[0], s[1])) for _ in range(s[2])]
    #zy = [zy.reshape((s[0], s[1])) for _ in range(s[2])]

    zx = zx.reshape((s[0], s[1]))
    zy = zy.reshape((s[0], s[1]))

    #zx = np.stack(zx, axis=2)
    #zy = np.stack(zy, axis=2)
    
    z = integrateFrankot(zx, zy)
    print(z)
    x = np.arange(0, s[0])
    y = np.arange(0, s[1])
    ys, xs = np.meshgrid(y, x)
    print(s)
    print(xs.shape)
    print(ys.shape)

    surface = [xs, ys, z]
    surface = np.stack(surface, axis=2)
    print(surface.shape)
    print(s)
    
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
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X = surface[:, :, 0]
    Y = surface[:, :, 1]
    Z = surface[:, :, 2]

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    plt.show()


    pass


if __name__ == '__main__':

    # Put your main code here
    
    # Q1.b
    '''
    light_dirs = [
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, 1]) / np.sqrt(3),
        np.array([-1, -1, 1]) / np.sqrt(3)
    ]
    res = (2160, 3840)
    pxSize = 7e-4  # 7 µm in meters
    rad = 0.0075  # 0.75 cm in meters
    center = np.array([0, 0, 0])
    for i in range(len(light_dirs)):
        image = renderNDotLSphere(center, rad, light_dirs[i], pxSize, res)
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.show()
    '''
    # Q1.c
    I, L, s = loadData()
    print(I.dtype)
    
    #Q1.d
    U, S, Vh = np.linalg.svd(I, full_matrices=False, compute_uv=True)
    print("Singular values of I:", S)

    # Q1.e
    B = estimatePseudonormalsCalibrated(I, L)
    print("shape of B:", B.shape)
    albedos, normals = estimateAlbedosNormals(B)

    # Q1.f
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imshow(albedoIm, cmap='gray')
    plt.show()

    plt.imshow(normalIm, cmap='rainbow')
    plt.show()

    # Q1.i
    surface = estimateShape(normals, s)
    plotSurface(surface)

    pass
