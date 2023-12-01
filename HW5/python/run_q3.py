import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

#plt.imshow(train_x[0].reshape(32, 32), cmap="Greys")
#plt.show()

test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 90
# pick a batch size, learning rate
batch_size = 16
learning_rate = 1e-3
#learning_rate = 1e-3 * 10
#learning_rate = 1e-3 * 0.1
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
'''
from mpl_toolkits.axes_grid1 import ImageGrid

# Q3.3 visualize initial weights here

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)

for i in range(hidden_size):
    grid[i].imshow(np.reshape(params['Wlayer1'][:, i], (32, 32))) 
    plt.axis('off')

plt.show()
'''
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
    total_loss /= train_x.shape[0]
    
    training_entropy_loss[itr] = total_loss 
    training_acc[itr] = total_acc

    valid_h1 = forward(valid_x, params, "layer1")
    valid_probs = forward(valid_h1, params, "output", softmax)
    val_loss, val_acc = compute_loss_and_acc(valid_y, valid_probs)

    validation_entropy_loss[itr] = val_loss / valid_x.shape[0]
    validation_acc[itr] = val_acc 
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

# run on validation set and report accuracy! should be above 75%
valid_acc = None
##########################
##### your code here #####
##########################
v1 = forward(valid_x, params, "layer1")
v2 = forward(v1, params, "output", softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, v2)

print('Validation accuracy: ',valid_acc)

# Q3.2 computing test accuracy
test_h1 = forward(test_x, params, "layer1")
test_probs = forward(test_h1, params, "output", softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, test_probs)

print('Test accuracy: ', test_acc)

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

'''
# Q3.2
plt.plot(epochs, training_acc)
plt.xlabel("epochs")
plt.ylabel("accuracy (%)")
plt.title(f'Training Accuracy with lr={learning_rate}')
plt.show()

plt.figure()
plt.plot(epochs, training_entropy_loss)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title(f'Training Loss with lr={learning_rate}')
plt.show()
'''
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
'''
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)

for i in range(hidden_size):
    grid[i].imshow(np.reshape(params['Wlayer1'][:, i], (32, 32))) 
    plt.axis('off')

plt.show()

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute confusion matrix here
##########################
##### your code here #####
##########################
for i in range(test_y.shape[0]):
    # get actual class of test set
    actual_class = np.argmax(test_y[i, :])

    # get predicted class from test set
    pred_class = np.argmax(test_probs[i, :])

    confusion_matrix[actual_class][pred_class] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
