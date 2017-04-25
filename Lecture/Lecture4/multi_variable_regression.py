# Not good Code

import tensorflow as tf

# Multi-Variable
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

# Y value
y_data = [152., 185., 180., 196., 142.]

# weight, shape = 1
w1 = tf.Variable(tf.random_normal([1]), name = "weight1")
w2 = tf.Variable(tf.random_normal([1]), name = "weight2")
w3 = tf.Variable(tf.random_normal([1]), name = "weight3")

# bias
b = tf.Variable(tf.random_normal([1]), name = "bias")

# hypothesis
hypothesis = w1 * x1_data + w2 * x2_data + w3 * x3_data + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# optimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(300001) :
    sess.run(train)

    if (step % 20 == 0) :
        print (step, sess.run(hypothesis), sess.run(w1), sess.run(w2), sess.run(w3), sess.run(b), sess.run(cost))
