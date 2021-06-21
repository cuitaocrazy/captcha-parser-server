import tensorflow as tf

a = tf.range(1*2*3*4)

a = tf.reshape(a, [4, 3, 2, 1])
print(a)
print(tf.transpose(a))