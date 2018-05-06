
from include.DataHandling import get_image_train2, get_image_CV2 
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.logging.set_verbosity(tf.logging.INFO)

len_train = 6000
len_CV = 600                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
gray = 0
img_size = 60
n_channels = 3
if gray:
    n_channels = 1

#lr = 1e-3
epochs = 1
batch_size = 64
n_classes = 2
drop_out_keep_prob = 0.8
dp = 1 - drop_out_keep_prob
reg = 0#1e-6

def print_params(lr, reg):
    print('lr: ', lr, 'reg', reg)

def log_parameters(lr, reg):
    arr = [len_train, len_CV, lr, epochs, batch_size, drop_out_keep_prob, reg]        
    np.savetxt('Results/params.txt', arr)
    print "Parameters logged"
    

def save(arr):
    print "Dumping in a file"
    with open('data.p', 'wb') as f:
        pickle.dump(arr, f)
    print "Saved"    

def load_data():
    print "Loading from pickle file"
    with open('data.pkl', 'rb') as f:
        [X_train, Y_train, X_CV, Y_CV] = pickle.load(f)
    print "Data Loaded"    
    return [X_train, Y_train, X_CV, Y_CV]    

#print "Initializing"


def get_data(len_train=len_train, len_CV=len_CV, gray=gray, n_channels=n_channels):
    X_train = np.zeros((2 * len_train, img_size, img_size, n_channels))
    X_CV = np.zeros((2 * len_CV, img_size, img_size, n_channels))
    Y_train = np.zeros((2*len_train, 2))
    Y_CV = np.zeros((2*len_CV, 2))

    print "Loading Train Data"
    for i in range(0, len_train):
        #print i+1 ,"/",len_train
        X_train[2*i] = get_image_train2(i, 'face', gray=gray)
        Y_train[2*i] = np.array([1,0])
        X_train[2*i+1] = get_image_train2(i, 'nonface', gray=gray)
        Y_train[2*i+1] = np.array([0,1])
    
    print "Loading Test Data"
    for i in range(0, len_CV):
        #print i+1, "/", len_CV
        X_CV[i] = get_image_CV2(i, 'face', gray=gray)
        X_CV[i + len_CV] = get_image_CV2(i, 'nonface', gray=gray)
        Y_CV[i] = np.array([1, 0])
        Y_CV[i + len_CV] = np.array([0, 1])


    X_train /= 255.0
    X_CV /= 255.0
     
    arr = [X_train, Y_train, X_CV, Y_CV] 
    return arr


[X_train, Y_train, X_CV, Y_CV] = get_data(len_train=len_train, len_CV=len_CV, gray=gray, n_channels=n_channels)

print "shuffling"
X_train, X_, Y_train, Y_ = train_test_split(X_train, Y_train, test_size=0, random_state=42)




##########################################################################################################################
##########################################################################################################################


#print "Build model"

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2d(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



x = tf.placeholder(tf.float32, [None, img_size*img_size])
x = tf.reshape(x, [-1, img_size, img_size, n_channels])
y = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)

W_conv1 = weight_variable([5, 5, n_channels, 16])
b_conv1 = bias_variable([16])
W_conv2 = weight_variable([5, 5, 16, 16])
b_conv2 = bias_variable([16])
W_fc1 = weight_variable([15*15*16, 128])
b_fc1 = bias_variable([128])


h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2d(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2d(h_conv2)

flattened = tf.reshape(h_pool2, [-1, 15*15*16])

h_fc1 = tf.nn.relu(tf.matmul(flattened, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, n_classes])
b_fc2 = bias_variable([n_classes])

y_ = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
prediction = tf.nn.softmax(y_)

total_batches = 2*len_train/batch_size

sess = tf.Session()


max_count = 10
for count in range(max_count):
    lr = np.random.uniform(0.000670, 0.000960)
    reg = np.random.uniform(0.004194, 0.020000)
    
    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_)) + \
                       reg*(tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + \
                               	       tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) )

    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    "Evaluate the model"
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    sess.run(tf.global_variables_initializer())

    train_acc_arr = []
    test_acc_arr = []
    cost_arr = []    
    epoch_arr = []
    
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batches):
            batch_x = X_train[i*batch_size : (i+1)*batch_size]        
            batch_y = Y_train[i*batch_size : (i+1)*batch_size] 
            _, c = sess.run([train_step, cross_entropy], 
                            feed_dict={ x: batch_x, y: batch_y, keep_prob: drop_out_keep_prob })
            avg_cost += c/total_batches
        cost_arr.append(avg_cost)    
        acc_train = sess.run(accuracy, feed_dict={x: X_train[0:1000], y: Y_train[0:1000], keep_prob:1.0})
        acc_test = sess.run(accuracy, feed_dict={x: X_CV, y: Y_CV, keep_prob:1.0})    
        print("cost:", '{0:.6f}'.format(avg_cost), "train_acc:", '{0:.6f}'.format(acc_train),\
                       "val_acc: ", '{0:.6f}'.format(acc_test), "lr: ", '{0:.6f}'.format(lr), "reg: ", '{0:.6f}'.format(reg))

    






