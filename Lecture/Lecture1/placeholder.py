import tensorflow as tf

# Create placeholder
placeholder_one = tf.placeholder(tf.float32)
placeholder_two = tf.placeholder(tf.float32)

add_placeholder = tf.add(placeholder_one, placeholder_two)

sess = tf.Session()

# Insult placeholder's value
feed_dict={placeholder_one : 3.2, placeholder_two : 1.4}

# Run 
print(sess.run(add_placeholder, feed_dict) )
