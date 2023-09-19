import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----

    # histogram is of length equal to the dictionary (e.g. K)
    hist = np.zeros(K)

    # count how often each words appears in the image
    for visual_word in range(K):
        hist[visual_word] = np.count_nonzero(wordmap == visual_word)
    
    # L1 normalization on histogram
    hist_norm = np.linalg.norm(hist, ord=1)
    hist = hist / hist_norm
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    height, width = wordmap.shape
    finest_layer_hist_matrix = np.zeros((2 ** L, 2 ** L, K))
    hist_all = []
    
    # creating the histogram of the finest layer as a 3d matrix (2^l x 2^l x K)
    # K = # of bag of words
    for cell_row in range(2 ** L):
        for cell_col in range(2 ** L):
            ith_cell_height = height // (2 ** L)
            ith_cell_width = width // (2 ** L)
            ith_cell = wordmap[cell_row * ith_cell_height : (cell_row + 1) * ith_cell_height, cell_col * ith_cell_width : (cell_col + 1) * ith_cell_width]
            cell_hist = get_feature_from_wordmap(opts, ith_cell)
            finest_layer_hist_matrix[cell_row, cell_col, :] = cell_hist
    
    # create overall histogram of all levels
    curr_layer_hist_matrix = finest_layer_hist_matrix
    for layer in range(L, -1, -1):
        # turn the current layer's histogram matrix into a normal histogram (vector)
        curr_layer_hist = curr_layer_hist_matrix.flatten(order='C')
            
        # apply weight to histogram
        if layer > 1:
            weight = 2 ** (layer - L - 1)
        else:
            weight = 2 ** (-L)
        curr_layer_hist = curr_layer_hist * weight
        
        # concatenate histogram
        hist_all = np.append(curr_layer_hist, hist_all)
        
        # normalize histogram after aggregation
        hist_norm = np.linalg.norm(hist_all, ord=1)
        hist_all = hist_all / hist_norm
            
        # create the previous layer's histogram matrix
        if layer > 0:
            prev_layer_hist_matrix = np.zeros((2 ** (layer - 1), 2 ** (layer - 1), K))
            rows, cols = prev_layer_hist_matrix.shape[0], prev_layer_hist_matrix.shape[1]
            for row_cell in range(rows):
                for col_cell in range(cols):
                    # histogram of previous layer can be aggregated from current layer
                    prev_layer_hist_matrix[row_cell, col_cell, :] = np.sum(curr_layer_hist_matrix[row_cell * 2 : (row_cell + 1) * 2, col_cell * 2 : (col_cell + 1) * 2, :], axis = 0).sum(axis = 0)
            curr_layer_hist_matrix = prev_layer_hist_matrix
            
    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    
    # create a wordmap that maps each pixel in the image to its closest word in the dictionary
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    
    # create histogram of visual words using spatial pyramid matching
    spatial_pyramid_hist = get_feature_from_wordmap_SPM(opts, wordmap)
    
    return spatial_pyramid_hist

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    # ----- TODO -----
    
    # create pool of workers
    pool = multiprocessing.Pool(n_worker)
    
    # extract histograms from every image
    spatial_histograms_list = [pool.apply_async(get_image_feature, args=(opts, join(data_dir, train_file), dictionary)) for train_file in train_files]
    
    features = [ar.get() for ar in spatial_histograms_list]
    
    # save data to build recognition system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
    pass

    ## example code snippet to save the learned system
    # np.savez_compressed(join(out_dir, 'trained_system.npz'),
    #     features=features,
    #     labels=train_labels,
    #     dictionary=dictionary,
    #     SPM_layer_num=SPM_layer_num,
    # )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    
    return np.sum(np.minimum(word_hist, histograms), axis=1)
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    pass

