import numpy as np
import os
import png
import tensorflow as tf
import preprocessing as pp

IMG_WIDTH = 640
IMG_HEIGHT = 400
IMG_DEPTH = 3
learning_rate = .0001
num_epochs = 100
batch_size = 50

frame_dirs = ["data/frames/frametest/"]#, "data/frames/frames2/"]
inputs_dirs = ["data/inputs/race1"]#, "data/inputs/race2"]

train_data = []
train_labels = []

for i in range(len(frame_dirs)):
    tdata, tlabels = pp.getData(frame_dirs[i], inputs_dirs[i])
    train_data.extend(tdata)
    train_labels.extend(tlabels)

print(train_data[0].shape)
train_data = np.array(train_data)
train_labels = np.array(train_labels)
num_samples = train_data.shape[0]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                            strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[batch_size, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH])
y_ = tf.placeholder(tf.float32, shape=[batch_size, 1])

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

#640x400x32 --> 320x200x32
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#320x200x32 --> 160x100x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([160*100*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 160*100*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cost = tf.reduce_sum(tf.pow(y_conv - y_, 2))/(2*num_samples)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Beginning optimization")
    for epoch in range(num_epochs):
        batch_x = np.array(train_data[epoch*batch_size:epoch*batch_size + batch_size])
        batch_x = np.reshape(batch_x, [batch_x.shape[0], IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH])
        batch_y = np.array(train_labels[epoch*batch_size:epoch*batch_size + batch_size])
        batch_y = np.reshape(batch_y, [batch_y.shape[0], 1])

        optimizer.run(feed_dict={x: batch_x, y_: batch_y})

        if epoch % 20 == 0:
            c = sess.run(cost, feed_dict={X: train_data, Y:train_labels})
            print("Epoch:", epoch, "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization finished")
