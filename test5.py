import cv2
import numpy as np
import sys
import tensorflow as tf 
read_img = cv2.imread("fengjing_6.jpeg")
read_img_1 = tf.constant(read_img, dtype = tf.float32)
read_img_1 = tf.reshape(read_img_1, [-1, 1228, 733, 3])
conv1 = tf.layers.conv2d(
	inputs = read_img_1,
	
        	filters = 1,
        	kernel_size = [240, 20],
        	padding = "same",
        	activation = tf.nn.relu
	)
bias_initializer=tf.global_variables_initializer()
sess = tf.Session()
sess.run(bias_initializer)
conv1_1 = sess.run(conv1)
#conv1_1 = tf.asarray(conv1_1, dtype=tf.float32)
#img_data_jpg = tf.image.decode_jpeg(conv1_1)  
#img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
#conv1_1.resize((480, 640, 3))
# print conv1_1[0]
# print "this is read_img", read_img
#cv2.imshow("test3", conv1_1[0])
#sess.run(img_data_jpg)
# cv2.namedWindow("test1")
read_img_2 = read_img[:, :, ::-1][:, ::-1]
read_img = np.hstack((read_img, read_img[:, :, ::-1][:, ::-1]))
cv2.imshow("test1", mat = read_img)
# cv2.imshow("test2", mat = read_img)
cv2.imwrite("save_1.jpg", read_img_2)
a = [[1, 0], [10, 1], [1, 1]]
b = tf.zeros_like(a)
b = sess.run(b)
print b
cv2.waitKey (0)
# for i in range(10):
# 	print i
