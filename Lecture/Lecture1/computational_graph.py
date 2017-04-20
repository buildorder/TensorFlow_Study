import tensorflow as tf

# Create Node
number_node_one = tf.constant(3.5, tf.float32)
number_node_two = tf.constant(4.2, tf.float32)

# It should same variable type
# Arguments : ( x, y, Name=Null )
add_node = tf.add(number_node_one, number_node_two)

sess = tf.Session()
print (sess.run(add_node))
