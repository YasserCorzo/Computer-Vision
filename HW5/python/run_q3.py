import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 80
# pick a batch size, learning rate
batch_size = 32
learning_rate = 1e-3
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, train_y.shape[1], params, 'output')

training_entropy_loss = np.zeros(max_iters)
training_acc = np.zeros(max_iters)

validation_entropy_loss = np.zeros(max_iters)
validation_acc = np.zeros(max_iters)

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        # forward prop
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        
        # calculate loss and add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc
        
        # backward propagation
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        delta3 = backwards(delta2,params,'layer1',sigmoid_deriv)
        
        # apply gradient
        params['Wlayer1'] = params['Wlayer1'] - (learning_rate * params['grad_Wlayer1'])
        params['blayer1'] = params['blayer1'] - (learning_rate * params['grad_blayer1'])
        params['Woutput'] = params['Woutput'] - (learning_rate * params['grad_Woutput'])
        params['boutput'] = params['boutput'] - (learning_rate * params['grad_boutput'])
        
    total_acc /= batch_num
    
    training_entropy_loss[itr] = total_loss
    training_acc[itr] = total_acc

    valid_forward_1 = forward(valid_x, params, "layer1")
    valid_forward_2 = forward(valid_forward_1, params, "output", softmax)
    val_loss, val_acc = compute_loss_and_acc(valid_y, valid_forward_2)

    validation_entropy_loss[itr] = val_loss
    validation_acc[itr] = val_acc 
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

# run on validation set and report accuracy! should be above 75%
valid_acc = None
##########################
##### your code here #####
##########################
v1 = forward (valid_x, params, "layer1")
v2 = forward(v1, params, "output", softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, v2)

print('Validation accuracy: ',valid_acc)

# plot validation and training accuracy
epochs = np.arange(0, max_iters)

plt.plot(epochs, validation_acc, label="validation accuracy")
plt.plot(epochs, training_acc, label="training accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy (%)")
plt.legend()
plt.title("Train & Validation Accuracy")
plt.show()

plt.figure()
plt.plot(epochs, validation_entropy_loss, label="validation average cross entropy")
plt.plot(epochs, training_entropy_loss, label="training average cross entropy")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.title("Train & Validation Loss")
plt.show()


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()