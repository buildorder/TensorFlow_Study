import tensorflow as tf

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Create Variable / 1 means Rank
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# set hypothesis
hypothesis = x_train * W + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# Launch the graph
sess = tf.Session()

# Initialize global variables
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    # Run train node
    sess.run(train)

    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b) )
