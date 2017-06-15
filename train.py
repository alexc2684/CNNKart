import numpy as np
import os
import png
import tensorflow as tf
import preprocessing as pp

IMG_WIDTH = 640
IMG_HEIGHT = 400
IMG_DEPTH = 3
LEARNING_RATE = .0001
NUM_EPOCHS = 100
BATCH_SIZE = 10

FRAMES_DIR = "data/frames"
INPUTS_DIR = "data/inputs"

train_data, train_labels = pp.load_train_data(FRAMES_DIR, INPUTS_DIR)

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

x = tf.placeholder(tf.float32, shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

x = tf.reshape(x, [-1, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH])
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
# print("FC Dropout", h_fc1_drop.shape)

W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

cost = tf.reduce_sum(tf.pow(y_conv - y_, 2))/(2*num_samples)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Beginning optimization")
    for epoch in range(NUM_EPOCHS):
        print("Epoch", epoch)
        start = epoch*BATCH_SIZE % num_samples
        end = start + BATCH_SIZE
        batch_x = train_data[start:end]
        batch_y = train_labels[start:end]
        batch_y = np.reshape(batch_y, (len(batch_y), 1))

        optimizer.run(feed_dict={x: batch_x, y_: batch_y})

        if epoch % 5 == 0:
            c = sess.run(cost, feed_dict={x: batch_x, y_:batch_y})
            print("Epoch:", epoch, "cost:", c)
    print("Optimization finished")

    saver = tf.train.Saver()
    saver.save(sess, "MLKart1.0", global_step=1000)
