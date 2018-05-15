import tensorflow as tf
from tensorboard import summary

a = tf.constant([2, 2], name = 'a')
b = tf.constant([2, 6], name = 'b')
x = tf.add(a, b, name = 'add')
with tf.Session() as sess: 
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))
    writer.close()
