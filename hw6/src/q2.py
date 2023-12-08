# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None
    # 3 x 7
    L = None

    # Perform SVD on the reshaped matrix
    U, S, Vt = np.linalg.svd(I, full_matrices=False)
    
    # to reduce to rank 3, set all singular values to 0 except top 3
    B = Vt[:3, :]
    L = U[:, :3].T

    print("shape of B hat:", B.shape)

    return B, L


if __name__ == "__main__":

    # Put your main code here
    I, L, s = loadData()

    B_hat, L_hat = estimatePseudonormalsUncalibrated(I)

    # Q2.f

    # u, v tilts plane
    u = 2 
    v = 0.5
    l = 0.5 # shrink or expands shape
    G = np.array(([1, 0, 0], [0, 1, 0], [u, v, l]))
    B_hat = np.linalg.inv(G) @ B_hat

    albedos, normals = estimateAlbedosNormals(B_hat)

    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    #plt.imshow(albedoIm, cmap='gray')
    #plt.show()

    #plt.imshow(normalIm, cmap='rainbow')
    #plt.show()

    #surface = estimateShape(normals, s)
    #plotSurface(surface)

    Nt = enforceIntegrability(B_hat, s)

    surface = estimateShape(Nt, s)
    plotSurface(surface)
    pass
