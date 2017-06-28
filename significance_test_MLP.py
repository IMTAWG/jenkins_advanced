import tensorflow as tf
import numpy as np
import cPickle as pickle
import time
import csv


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, mean=0, stddev=0.01))


def model(X, w_h, w_h2, w_o, b_h, b_h2, b_o, p_keep_input, p_keep_hidden):

    # RELU layer
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.tanh(tf.add(tf.matmul(X, w_h), b_h))

    # Sigmoid layer
    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.add(tf.matmul(h, w_h2), b_h2))

    # Readout layer
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    return tf.matmul(h2, w_o) + b_o


def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)


def xrange(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    else:
        return([])

for run in range(1, 2, 1):
    """Loading test and training data"""

    trX = pickle.load(open("unlabelled_svd_train.p", "rb"))
    trY = csv_to_numpy_array("labels_svd_train_2col.csv", delimiter=',')

    teX = pickle.load(open("unlabelled_svd_test.p", "rb"))
    teY = csv_to_numpy_array("labels_svd_test_2col.csv", delimiter=',')


    """Placeholders"""
    X = tf.placeholder("float", [None, 50]) # create symbolic variables
    Y = tf.placeholder("float", [None, 2])

    w_h = init_weights([50, 25]) # create symbolic variables
    w_h2 = init_weights([25, 25])
    w_o = init_weights([25, 2])

    b_h = init_weights([25]) # create symbolic variables
    b_h2 = init_weights([25])
    b_o = init_weights([2])

    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(X, w_h, w_h2, w_o, b_h, b_h2, b_o, p_keep_input, p_keep_hidden)

    learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                              global_step=1,
                                              decay_steps=trX.shape[0],
                                              decay_rate=0.95,
                                              staircase=True)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    # cost = tf.nn.l2_loss(py_x - Y, name="squared_error_cost")
    # train_op = tf.train.RMSPropOptimizer(0.0008, 0.9).minimize(cost)
    train_op = tf.train.RMSPropOptimizer(learningRate).minimize(cost)
    # train_op = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    """Running the program"""
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        time0 = time.time()
        for i in range(0, 2700):
            # Splitting the input into slices of 128 for faster training
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX) + 1, 128)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                              p_keep_input: 0.9, p_keep_hidden: 0.5})

            if i % 10 == 0:
                print 'Step: %d' % i
                print(np.mean(np.argmax(teY, axis=1) == sess.run(predict_op,
                                                                 feed_dict={X: teX, p_keep_input
                                                                 : 1.0, p_keep_hidden: 1.0})))
        with open('MLP_tanrelu_run%s.csv' % run,
                  'wb') as csvfile:  # Creates .csv file to store the inputted tweet.
            w = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
            w.writerow(['Final Accuracy:%f' % (np.mean(np.argmax(teY, axis=1) ==
                                                       sess.run(predict_op, feed_dict={X: teX, p_keep_input
                                                       : 1.0, p_keep_hidden: 1.0})))])

    time1 = time.time()
    sess.close()

print 'Total time: %f' % (time1 - time0)
