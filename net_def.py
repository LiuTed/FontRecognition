import tensorflow as tf
import numpy as np

def bn_layer(inputs, name):
	with tf.name_scope(name):
		mean, var = tf.nn.moments(inputs, axes = [0,1,2], name='moments')
		offset = tf.Variable(tf.zeros_like(mean), name = 'offset')
		scale = tf.Variable(tf.ones_like(mean), name = 'scale')
		bn = tf.nn.batch_normalization(inputs, mean, var, offset, scale, tf.constant(1e-3), name = 'batch_normalization')
		return bn

def conv_layer(inputs, fsize, channel_out, name, stride = [1,1,1,1],
	padding = "SAME", use_bn = False, activate = tf.nn.relu6):
	filter_size = [fsize, fsize, inputs.get_shape()[-1], channel_out]
	filter_size = tf.TensorShape(filter_size)
	with tf.name_scope(name):
		f = tf.Variable(tf.truncated_normal(filter_size, stddev = 0.1), name = "filter")
		conv = tf.nn.conv2d(inputs, f, stride, padding, name = "convolution")
		if use_bn:
			conv = bn_layer(conv, "bn")
		if activate is not None:
			return activate(conv, name = "activation")
		else:
			return conv

def dense_layer(inputs, out, name, use_bias = True):
	with tf.name_scope(name):
		wshape = [inputs.get_shape()[-1], out]
		wshape = tf.TensorShape(wshape)
		weight = tf.Variable(tf.truncated_normal(wshape, stddev = 0.1), name = "weights")
		dense = tf.matmul(inputs, weight)
		if use_bias:
			bias = tf.Variable(tf.truncated_normal([out], stddev = 0.1), name = "bias")
			dense = tf.nn.bias_add(dense, bias)
		return dense


def residue_block(inputs, channel_out, name, half_size = False):
	with tf.name_scope(name):
		if half_size:
			stride = [1,2,2,1]
			shortcut = conv_layer(inputs, 2, channel_out, "shortcut", stride = stride)
		else:
			stride = [1,1,1,1]
			if inputs.get_shape().as_list()[-1] != channel_out:
				shortcut = conv_layer(inputs, 1, channel_out, "shortcut", stride = stride)
			else:
				shortcut = inputs
		
		conv1 = conv_layer(inputs, 3, channel_out, "convolution1", stride=stride)
		# conv2 = conv_layer(conv1, 3, channel_out, "convolution2", stride=[1,1,1,1], activate = None)
		conv2 = conv1
		return tf.nn.relu6(conv2 + shortcut, name = "activation")

def resnet(x, nconvs, name):
	assert nconvs[0] == 1, "conv1 should only contain one convolution layer"
	with tf.name_scope(name):
		conv1 = conv_layer(x, 5, 32, "conv1", stride = [1,2,2,1])
		pool1 = tf.nn.max_pool(conv1, [1,3,3,1], [1,2,2,1], padding = "SAME", name = "pool1")
		conv = pool1
		depth = conv.get_shape().as_list()[-1]
		for i in range(1, len(nconvs)):
			for j in range(nconvs[i]):
				conv = residue_block(conv, depth * 2, "conv%d_%d"%(i+1,j+1), half_size = (i > 1 and j == 0))
			depth *= 2

		return conv

def resnet_18(x, n_class):
	res = resnet(x, [1, 2, 2, 2, 2], "ResNet-18")
	res = tf.reduce_mean(res, [1, 2])
	return dense_layer(res, n_class, "dense")

def lenet(x, n_class):
    x = tf.image.resize_bilinear(x, [32, 32])
    c1 = conv_layer(x, 5, 6, "C1", padding = "VALID")
    s2 = tf.nn.max_pool(c1, [1,2,2,1], [1,2,2,1], "SAME", name = "S2")
    c3 = conv_layer(s2, 5, 16, "C3", padding = "VALID")
    s4 = tf.nn.max_pool(c3, [1,2,2,1], [1,2,2,1], "SAME", name = "S4")
    c5 = conv_layer(s4, 5, 120, "C5", padding = "VALID")
    c5 = tf.reshape(c5, [-1, 120])
    f6 = dense_layer(c5, n_class, "F6")
    return f6

