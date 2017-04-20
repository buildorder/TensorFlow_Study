import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Use placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Create Variable / 1 means Rank
W = tf.Variable(tf.random_normal([1]), name='weight')

# set hypothesis
hypothesis = X * W

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
learning_rate = 0.1
gradient = tf.reduce_mean((X * W - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# Launch the graph
sess = tf.Session()
# Initialize global variables
sess.run(tf.global_variables_initializer())

for step in range(201) :
    sess.run(update, feed_dict = {X : x_data, Y : y_data})
    print (step, sess.run(W), sess.run(cost, feed_dict = {X : x_data, Y : y_data}))
