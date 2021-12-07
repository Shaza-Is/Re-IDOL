import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class ReOrientLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    
    q = tf.slice(y_true,
                    begin=[0,0],
                    size=[-1,4])  

    q_est = tf.slice(y_pred,
                    begin=[0,0],
                    size=[-1,4])  


    s_00 = tf.math.exp(tf.reshape(tf.slice(y_pred,  
           begin=[0,4],
           size=[-1,1]), [-1]))
    s_11 = tf.math.exp(tf.reshape(tf.slice(y_pred,  
           begin=[0,5],  
           size=[-1,1]), [-1]))
    s_22 = tf.math.exp(tf.reshape(tf.slice(y_pred, 
           begin=[0,6],
           size=[-1,1]), [-1]))
    s_01 = tf.math.multiply(tf.reshape(tf.slice(y_pred,
           begin=[0,7],  
           size=[-1,1]), [-1]), tf.math.sqrt(tf.math.multiply(s_00, s_11)))
    s_02 = tf.math.multiply(tf.reshape(tf.slice(y_pred, 
           begin=[0,8],    
           size=[-1,1]), [-1]), tf.math.sqrt(tf.math.multiply(s_00, s_22)))
    s_12 = tf.math.multiply(tf.reshape(tf.slice(y_pred,
           begin=[0,9],  
           size=[-1,1]), [-1]), tf.math.sqrt(tf.math.multiply(s_11, s_22)))

           
    sig0 = tf.stack([s_00, s_01, s_02], axis=1)
    sig1 = tf.stack([s_01, s_11, s_12], axis=1)
    sig2 = tf.stack([s_02, s_12, s_22], axis=1)
    sig = tf.stack([sig0, sig1, sig2], axis=2)
    i = tfg.quaternion.inverse(q_est)
    mult_ = tfg.quaternion.multiply(i, q)
    log_ = tf.slice(mult_,
                    begin=[0,0],
                    size=[-1,3])
    omega=tf.slice(mult_,
              begin=[0,3],
              size=[-1,1])

 
    log_ = tf.linalg.normalize(log_, ord='euclidean', axis=None, name=None)
    atan_ = tf.atan(tf.divide(log_[1], omega))
    v = log_[0]
    z = tf.zeros_like(v)
    v = tf.where(tf.equal(v, z), tf.zeros_like(v), v)

    
    delta = tf.math.scalar_mul(2.0, tf.multiply(v, atan_))                  
    delta = tf.expand_dims(delta, -1)
    
    e = tf.eye(3, batch_shape=[64])

    sig_inv = tf.linalg.inv(sig) 
    m_ = tf.matmul(sig_inv, delta)
    m = tf.matmul(delta, m_, transpose_a=True)

    #l_s = tf.linalg.logdet(sig)
    #l = tf.math.scalar_mul(0.5, m)

    l_s = tf.linalg.logdet(sig)
    ls_without_nans = tf.where(tf.math.is_nan(l_s), tf.zeros_like(l_s), l_s)
    l = tf.math.scalar_mul(0.5, m) + tf.math.abs(tf.math.scalar_mul(0.5, ls_without_nans)) 

    l = tf.cast(l, tf.float32)
    return tf.reduce_mean(l)


class MyLossP(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_pred - y_true)**2, axis=-1))

def quat_diff(y_true, y_pred):
  q = tf.slice(y_true,
                    begin=[0,0],
                    size=[-1,4])  



  q_est = tf.slice(y_pred,
                    begin=[0,0],
                    size=[-1,4])  
  d = tfg.quaternion.relative_angle(q_est, q)
  # Replace all NaN values with 0.0.
  d_without_nans = tf.where(tf.math.is_nan(d), tf.zeros_like(d), d)
  return tf.cast(d_without_nans, tf.float32)

quat_metric = tf.keras.metrics.MeanMetricWrapper(fn=quat_diff, name='metric_quat_diff')
