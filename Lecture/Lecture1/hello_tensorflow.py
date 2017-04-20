import tensorflow as tf

# Create Node
hello = tf.constant("Hello, TensorFlow!")

# Create Session
sess = tf.Session()

# Run Session
print (sess.run(hello))
