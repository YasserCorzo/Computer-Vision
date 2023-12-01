import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hidden')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, train_x.shape[1], params, 'output')

# create list of keys (before addition of momentum)
keys = []
for k in params.keys():
    keys.append(k)

# initalize zero-momentum accumulators for each parameter
for k in keys:
    params['Mw_' + k] = np.zeros(params[k].shape)

training_entropy_loss = np.zeros(max_iters)

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################

        # forward prop
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        probs = forward(h3, params, 'output', sigmoid)
        
        # calculate loss and add loss to epoch totals
        loss = np.sum(np.square(xb - probs))
        total_loss += loss
        
        # backward propagation
        delta1 = 2 * (probs - xb)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'hidden2', relu_deriv)
        delta4 = backwards(delta3, params, 'hidden', relu_deriv)
        delta5 = backwards(delta4, params, 'layer1', relu_deriv)
        
        # apply momentum
        for k in params.keys():
            
            # only need to update momentum rules (Mw_*) and corresponding weigh (W*) (not grad_* or cache_*)
            if '_' in k: 
                continue

            # update rules M_w
            params['Mw_' + k] = (0.9 * params['Mw_' + k]) - (learning_rate * params['grad_' + k])

            # update weights
            params[k] += params['Mw_' + k]

    total_loss /= batch_num

    training_entropy_loss[itr] = total_loss

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

plt.plot(range(max_iters), training_entropy_loss)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Training Loss")
plt.show()
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################
valid_y = valid_data['valid_labels']

# select 5 classes from total 36 classes
classes = np.random.choice(range(36), size=(5), replace=False)
print("classes:", classes)

# forward prop
h1 = forward(valid_x, params, 'layer1', relu)
h2 = forward(h1, params, 'hidden', relu)
h3 = forward(h2, params, 'hidden2', relu)
probs = forward(h3, params, 'output', sigmoid)
print(valid_x.shape)
# retrieve 2 inputs for each class and their conversions
x = []
l_recon = []
b_size = valid_x.shape[0] // batch_size
for c in classes:
    batch_input = valid_x[c*b_size : (c+1)*b_size]
    batch_probs = probs[c*b_size : (c+1)*b_size]
    print("batch input size:", batch_input.shape)
    random_i = np.random.choice(range(b_size), size=(2), replace=False)
    random_inputs_class_c = batch_input[random_i]
    random_probs_class_c = batch_probs[random_i]
    x.append(random_inputs_class_c)
    l_recon.append(random_probs_class_c)

# 5 (classes) x 2 (images per class) x dim (32*32)
x = np.array(x)

# 5 (classes) x 2 (images per class) x dim (32*32)
l_recon = np.array(l_recon)

fig, axes = plt.subplots(len(classes), 4)


for i in range(x.shape[0]):
    # retrieve original validation images and plot it
    axes[i, 0].imshow(x[i, 0, :].reshape(32, 32).T)
    axes[i, 2].imshow(x[i, 1, :].reshape(32, 32).T)

    axes[i, 0].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    axes[i, 2].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    
    # retrieve reconstructed images and plot it
    axes[i, 1].imshow(l_recon[i, 0, :].reshape(32, 32).T)
    axes[i, 3].imshow(l_recon[i, 1, :].reshape(32, 32).T)

    axes[i, 1].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    axes[i, 3].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

plt.show()

# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################

total_PSNR = 0
for i in range(valid_x.shape[0]):
    total_PSNR += peak_signal_noise_ratio(valid_x[i, :], probs[i, :])

avg_PSNR = total_PSNR / valid_x.shape[0]

print("average PSNR:", avg_PSNR)