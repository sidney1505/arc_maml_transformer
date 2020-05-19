import tensorflow as tf


categorical_crossentropy = lambda y_true, y_pred: tf.keras.backend.mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

# y = tf.boolean_mask(tf.ones([1,11]), tf.reduce_any(tf.not_equal(tf.math.argmax(tf.constant([[0,0,0,0,0,0,0,0,0,0,1]],dtype=tf.float32),-1), 10), -1))
def categorical_crossentropy_masked(y_true, y_pred):
	mask = tf.tile(tf.expand_dims(tf.not_equal(tf.math.argmax(y_true,-1), 10), -1), [1,1,1,11])
	y_true_masked = y_true * tf.cast(mask, tf.float32)
	y_pred_masked = y_pred * tf.cast(mask, tf.float32)
	return tf.keras.backend.mean(tf.keras.losses.categorical_crossentropy(y_true_masked, y_pred_masked))


def focal_loss(y_true, y_pred):
	gamma = 2.0
	alpha = 0.25
	pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
	return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def focal_loss_masked(y_true, y_pred):
	gamma = 2.0
	alpha = 0.25
	mask = tf.tile(tf.expand_dims(tf.not_equal(tf.math.argmax(y_true,-1), 10), -1), [1,1,1,11])
	y_true_masked = y_true * tf.cast(mask, tf.float32)
	y_pred_masked = y_pred * tf.cast(mask, tf.float32)
	pt_1 = tf.where(tf.equal(y_true_masked, 1), y_pred_masked, tf.ones_like(y_pred_masked))
	pt_0 = tf.where(tf.equal(y_true_masked, 0), y_pred_masked, tf.zeros_like(y_pred_masked))
	return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
