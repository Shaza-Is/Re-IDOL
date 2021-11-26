import tensorflow as tf

class ReOrientLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        quaternion = tf.slice(y_true, begin=[0,0], size=[4,64])
        quaternion_estimate = tf.slice(y_pred, begin=[0,0], size=[4,64])

        