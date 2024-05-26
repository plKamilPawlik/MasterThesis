import tensorflow as tf

class CauchyNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, amplify: float, epsilon: float, clip=(0, 1), **kwargs):
        super(CauchyNoiseLayer, self).__init__()
        self.clip = clip
        self.amplify = amplify
        self.epsilon = epsilon

    def call(self, inputs, training=None):
        r1 = tf.random.normal(shape=tf.shape(inputs), mean=0, stddev=1)
        r2 = tf.random.normal(shape=tf.shape(inputs), mean=0, stddev=1)
        
        if training:
            return tf.clip_by_value(
                inputs + self.amplify*(r1 / r2),
                self.clip[0],
                self.clip[1],
            )
        else:
            return tf.clip_by_value(
                inputs + self.amplify*(r1 / r2) * self.epsilon,
                self.clip[0],
                self.clip[1],
            )

class NormalNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, mean: float, stddev: float, epsilon: float, clip=(0, 1), **kwargs):
        super(NormalNoiseLayer, self).__init__()
        self.clip = clip
        self.mean = mean
        self.stddev = stddev
        self.epsilon = epsilon

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=self.mean,
                stddev=self.stddev
            )
        else:
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=self.mean,
                stddev=self.stddev
            )
            
        return tf.clip_by_value(inputs + noise, self.clip[0], self.clip[1])

class UniformNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, minval: float, maxval: float, epsilon: float, clip=(0, 1), **kwargs):
        super(UniformNoiseLayer, self).__init__()
        self.clip = clip
        self.minval = minval
        self.maxval = maxval
        self.epsilon = epsilon
        
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.uniform(
                shape=tf.shape(inputs),
                minval=self.minval * self.epsilon,
                maxval=self.maxval * self.epsilon,
            )
        else:
            noise = tf.random.uniform(
                shape=tf.shape(inputs),
                minval=self.minval,
                maxval=self.maxval,
            )
            
        return tf.clip_by_value(inputs + noise, self.clip[0], self.clip[1])
