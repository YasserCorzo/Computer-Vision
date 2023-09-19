import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
from tempfile import TemporaryFile

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----
    
    # if image is in greyscale, duplicate them into 3 channels
    if img.ndim != 3:
        img = skimage.color.gray2rgb(img)
    
    # convert image into Lab color space
    img = skimage.color.rgb2lab(img)
    
    # retrieve LAB channels
    L = img[:, :, 0]
    A = img[:, :, 1]
    B = img[:, :, 2]
    
    filter_responses = []
    
    # convolve all filters with the image with all scales. Scales are 1 and 2. Filters are:
    # (1) Gaussian, (2) Laplacian of Gaussian, (3) derivative of Gaussian in the x direction, and (4) derivative of Gaussian in the y direction
    for scale in filter_scales:
        # apply filter (1) on every channel
        l_filter_response = scipy.ndimage.gaussian_filter(L, sigma=scale, mode='nearest')
        a_filter_response = scipy.ndimage.gaussian_filter(A, sigma=scale, mode='nearest')
        b_filter_response = scipy.ndimage.gaussian_filter(B, sigma=scale, mode='nearest')
        filter_responses.append(l_filter_response)
        filter_responses.append(a_filter_response)
        filter_responses.append(b_filter_response)
        # apply filter (2)
        l_filter_response = scipy.ndimage.gaussian_laplace(L, sigma=scale, mode='nearest')
        a_filter_response = scipy.ndimage.gaussian_laplace(A, sigma=scale, mode='nearest')
        b_filter_response = scipy.ndimage.gaussian_laplace(B, sigma=scale, mode='nearest')
        filter_responses.append(l_filter_response)
        filter_responses.append(a_filter_response)
        filter_responses.append(b_filter_response)
        # apply filter (3)
        l_filter_response = scipy.ndimage.gaussian_filter(L, sigma=scale, order=1, axes=1, mode='nearest')
        a_filter_response = scipy.ndimage.gaussian_filter(A, sigma=scale, order=1, axes=1, mode='nearest')
        b_filter_response = scipy.ndimage.gaussian_filter(B, sigma=scale, order=1, axes=1, mode='nearest')
        filter_responses.append(l_filter_response)
        filter_responses.append(a_filter_response)
        filter_responses.append(b_filter_response)
        # apply filter (4)
        l_filter_response = scipy.ndimage.gaussian_filter(L, sigma=scale, order=1, axes=0, mode='nearest')
        a_filter_response = scipy.ndimage.gaussian_filter(A, sigma=scale, order=1, axes=0, mode='nearest')
        b_filter_response = scipy.ndimage.gaussian_filter(B, sigma=scale, order=1, axes=0, mode='nearest')
        filter_responses.append(l_filter_response)
        filter_responses.append(a_filter_response)
        filter_responses.append(b_filter_response)
        
    filter_responses = np.stack(filter_responses, axis=2)
    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    train_file, opts = args
    data_dir = opts.data_dir
    alpha = opts.alpha
    
    # read the image
    img_path = join(data_dir, train_file)   
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    
    # extract filter response from image
    filter_response = extract_filter_responses(opts, img)
    
    # extract alpha points from filter response
    x = np.random.randint(0, img.shape[1], size=alpha)
    y = np.random.randint(0, img.shape[0], size=alpha)
    points = np.vstack((x, y)).T
    alpha_points = filter_response[points[:, 1], points[:, 0], :]
    return alpha_points
    

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    
    # ----- TODO -----
    
    # initalize aggregated filter responses from every image (alpha*T x 3F)
    filter_bank_num = len(opts.filter_scales) * 4
    filter_responses = np.zeros((alpha * len(train_files), 3 * filter_bank_num))
    
    # Create pool of workers
    pool = multiprocessing.Pool(n_worker)
    
    # collect coordinates of alpha random points from every image in the training set
    args = [(train_file, opts) for train_file in train_files]
    alpha_points_list = list(pool.map(compute_dictionary_one_image, args))
    
    # add alpha points from every image onto the filter responses matrix
    for (file_num, path) in enumerate(train_files):
        alpha_points = alpha_points_list[file_num]
        filter_responses[file_num * alpha: (file_num + 1) * alpha, :] = alpha_points

    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    pass

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    
    # initialize wordmap where each pixel is assigned the closest visual word
    # of the filter response at the respective pixel
    wordmap = np.zeros((img.shape[0], img.shape[1]))
    
    # extract filter response of img
    filter_responses = extract_filter_responses(opts, img)
    rows, cols = filter_responses.shape[0], filter_responses.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            # retrieve vector from pixel in the filter response (1 x 3F)
            filter_responses_vec = np.array([filter_responses[i, j, :]])
            
            # calculate distance between vector from a pixel in the filter response to visual words in dictionary
            dist_to_visual_words = scipy.spatial.distance.cdist(filter_responses_vec, dictionary, metric='euclidean')
            
            # retrieve closest visual word
            closest_visual_word = np.argmin(dist_to_visual_words)
            wordmap[i, j] = closest_visual_word

    return wordmap

