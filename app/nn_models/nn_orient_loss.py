import tensorflow as tf

class ReOrientLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_pred - y_true)**2, axis=-1))
        