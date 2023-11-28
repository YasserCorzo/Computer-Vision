import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    
    # use normalized initialization to initialize the values of the weights
    min_dist = -(np.sqrt(6) / np.sqrt(in_size + out_size))
    max_dist = np.sqrt(6) / np.sqrt(in_size + out_size)
    W = np.random.uniform(min_dist, max_dist, size=(in_size, out_size))
    b = np.zeros(out_size)

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    
    res = 1.0 / (1.0 + np.exp(-1.0 * x))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################

    pre_act = (X @ W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    c = np.max(x, axis=1)
    
    # apply translation to x
    x_translate = x - c.reshape(-1, 1)
    
    # apply exp to translated x
    res = np.exp(x_translate)
    
    res = res / np.sum(res, axis=1).reshape(-1, 1)
    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################
    
    # retrieve number of examples
    N = len(y)
    
    # compute loss
    loss = -np.sum(np.sum(y * np.log(probs), axis=1))
    
    # compute the argmax across every row for probs and y and compare them
    argmax_truth_arr = np.argmax(y, axis=1) == np.argmax(probs, axis=1)
    
    # calculate accuracy
    acc = np.sum(argmax_truth_arr) / N
    
    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################
    
    # derivative of activation
    d_post_act = activation_deriv(post_act)

    d_b = np.ones_like(b)
    d_b = d_b[np.newaxis, :]
    
    grad_W = ((delta * d_post_act).T @ X).T

    grad_b = np.sum(delta * d_post_act, axis=0)
    #grad_b = ((delta.T @ d_post_act) @ d_b.T).T
    #grad_b = grad_b[0]

    # N x 1
    grad_X = ((delta * d_post_act)) @ W.T
    
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################
    
    # combine inputs and their relative ground truth outputs 
    # (x & y respectively) into one matrix
    D = np.hstack((x, y))
    
    # randomly shuffle dataset
    np.random.shuffle(D)
    
    # split dataset into batches
    for i in range(0, len(D), batch_size):
        batch = D[i:i + batch_size, :]
        
        # split batch into features and output
        batch_x = batch[:, :x.shape[1]]
        batch_y = batch[:, x.shape[1]:]
        
        # add split features and output of batch to batch list
        batches.append((batch_x, batch_y))
        
    return batches
