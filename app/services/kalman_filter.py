# class handles Kalman filter for quaternion estimation based on gyroscope and nn_orient and manifold encapsulation
import tensorflow as tf

class KalmanFilter:
    def __init__(self):
        self.Q = 0.005*tf.math.eye(3)
        self.R = tf.math.diag([1, 1, 1])**2             # nn_orientation covariance
        self.x = tf.math.array([[0.], [0.], [0.], [0.]]) # state vector "quaternion"
        self.P = tf.math.eye(3)                         # covariance of Kf state vector
        self.w = tf.math.array([[0.], [0.], [0.]])      # gyroscope measurement
        self.q = tf.math.array([[0.], [0.], [0.], [0.]]) # nn_orientation measurement
        self.dt = 0.01                                  # time step
        self.I = tf.math.eye(3)                         # identity matrix
        self.B = tf.math.eye(3)                         # Quaternion prediction from gyroscope


    def predict(self, w):
        # predict state
        self.B = tf.math.array([[0.], [0.], [0.]])
        self.x = self.x + self.dt*tf.math.dot(self.B, w)
        self.P = self.P + self.Q

    def update(self, q, R):
        # update state
        S = self.P + R 
        K = tf.math.dot(self.P, tf.linalg.inv(S))
        self.x = self.x # Boxipls K * (q (Boximinus) self.x) 
        self.P = self.P - tf.math.dot(tf.math.dot(K, self.B), self.P)
        

