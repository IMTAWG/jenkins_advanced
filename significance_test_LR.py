from __future__ import division
import tensorflow as tf
import numpy as np
import time
import cPickle as pickle
import csv


def xprint(string):
    message = tf.constant(string)
    sess = tf.Session()
    print(sess.run(message))


def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)


# def model(X, weights, bias):
#
#     # Logistic Regression PREDICTION ALGORITHM i.e. FEEDFORWARD ALGORITHM
#
#     apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
#
#     add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
#
#     # Activation function
#     activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
#     return activation_OP

def model(X, weights, bias):

    X = tf.nn.dropout(X, 1.0) # No dropout used
    activate = tf.nn.sigmoid(tf.add(tf.matmul(X, weights, name="apply_weights"), bias, name="add_bias"))

    return activate


"""Loading test and training data"""

trainX = pickle.load(open("unlabelled_svd_train.p", "rb"))
trainY = csv_to_numpy_array("labels_svd_train_2col.csv", delimiter=',')
xprint('\nTraining data')
xprint(trainX.shape)
xprint(trainY.shape)

testX = pickle.load(open("unlabelled_svd_test.p", "rb"))
testY = csv_to_numpy_array("labels_svd_test_2col.csv", delimiter=',')
xprint('\nTest data')
xprint(testX.shape)
xprint(testY.shape)

"""Program parameters"""

numFeatures = trainX.shape[1]
xprint(numFeatures)

# numLabels = number of authors
numLabels = trainY.shape[1]
xprint(numLabels)

for run in range(1, 21, 1):
    """Learning rate for optimiser"""

    # numEpochs is the number of iterations
    numEpochs = 270000
    learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                              global_step=1,
                                              decay_steps=trainX.shape[0],
                                              decay_rate=0.95,
                                              staircase=True)
    """Training Queue"""

    q = tf.FIFOQueue(capacity=5, dtypes=tf.float32) # enqueue batches the size of the num features
    enqueue_op = q.enqueue(trainX)
    numberOfThreads = 1
    qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
    tf.train.add_queue_runner(qr)
    X = q.dequeue()  # It replaces our input placeholder
    # We can also compute y_true right into the graph now
    yGold = trainY

    """Test Set Placeholders"""

    # 'None' means no limit on number of rows
    XTest = tf.placeholder(tf.float32, [None, numFeatures])
    # yGold are the correct answers, rows have [1,0] for Trump or [0,1] for Hillary
    yTest = tf.placeholder(tf.float32, [None, numLabels])

    """Variable initialisation"""

    weights = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                           mean=0,
                                           stddev=0.01,
                                           # stddev=(np.sqrt(6 / numFeatures + numLabels + 1)),
                                           name="weights"))

    bias = tf.Variable(tf.random_normal([1, numLabels],
                                        mean=0,
                                        stddev=0.01,
                                        # stddev=(np.sqrt(6 / numFeatures + numLabels + 1)),
                                        name="bias"))

    """Tensorflow operations for prediction"""

    # INITIALIZE our weights and biases
    init_OP = tf.global_variables_initializer()

    activation_OP = model(X, weights, bias)

    """Tensorflow operation for error measurement"""

    # Loss function COST FUNCTION i.e. MEAN SQUARED ERROR
    cost_OP = tf.nn.l2_loss(activation_OP - yGold, name="squared_error_cost")
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activation_OP, labels=yGold))

    error = cost_OP
    # error = cross_entropy
    """Tensorflow operation for optimisation"""

    # OPTIMIZATION ALGORITHM i.e. GRADIENT DESCENT
    rate = learningRate
    # rate = 0.05
    # rate = 0.01

    training_OP = tf.train.GradientDescentOptimizer(rate).minimize(error)

    epoch_values = []
    accuracy_values = []
    cost_values = []

    """Running the program"""

    # Create a tensorflow session
    sess = tf.Session()

    # Initialize all tensorflow variables
    sess.run(init_OP)

    # ... add the coordinator, ...
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    """Evaluation of the model"""
    # Compares the predicted label with the actual label held in yGold
    correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))
    # False is 0 and True is 1, what was our average?
    accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
    # Summary op for regression output
    activation_summary_OP = tf.summary.histogram("output", activation_OP)
    # Summary op for accuracy
    accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
    # Summary op for cost
    cost_summary_OP = tf.summary.scalar("cost", cost_OP)
    # Summary ops to check how variables (W, b) are updating after each iteration
    weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
    biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))
    # Merge all summaries
    all_summary_OPS = tf.summary.merge_all()
    # Summary writer
    writer = tf.summary.FileWriter("summary_logs", sess.graph)

    # Initialize reporting variables
    cost = 0
    diff = 1

    # Training epochs
    start = time.time()
    for i in range(numEpochs):
        if i > 1 and diff < .0001:
            print("change in cost %g; convergence." % diff)
            break
        else:
            # Run training step
            step = sess.run(training_OP)
            # Report occasional stats
            if i % 10 == 0:
                # Add epoch to epoch_values
                epoch_values.append(i)
                # Generate accuracy stats on test data
                summary_results, train_accuracy, newCost = sess.run(
                    [all_summary_OPS, accuracy_OP, cost_OP])

                # Add accuracy to live graphing variable
                accuracy_values.append(train_accuracy)
                # Add cost to live graphing variable
                cost_values.append(newCost)
                # Write summary stats to writer
                writer.add_summary(summary_results, i)
                # Re-assign values for variables
                diff = abs(newCost - cost)
                cost = newCost

                # generate print statements
                print("step %d, training accuracy %g" % (i, train_accuracy))
                print("step %d, cost %g" % (i, newCost))
                print("step %d, change in cost %g" % (i, diff))

    with open('LR_run%s.csv' % run,
              'wb') as csvfile:  # Creates .csv file to store the inputted tweet.
        w = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        w.writerow(["final accuracy on test set:%s, Steps:%d" % (str(sess.run(accuracy_OP,
                                                          feed_dict={XTest: testX,
                                                                     yTest: testY})),i)])
    coord.request_stop()
    coord.join(threads)
    sess.close()

