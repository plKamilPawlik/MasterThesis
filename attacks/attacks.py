import tensorflow as tf

def FastFool(model, images, labels, epsilon: float, clip=(0, 1)):
    calculateLoss = tf.keras.losses.sparse_categorical_crossentropy
    calculatePool = tf.keras.Sequential([
        tf.keras.Input((28, 28, 1)),
        tf.keras.layers.Conv2D(
            dtype=tf.float32,
            padding='same',
            filters=1,
            kernel_size=3,
            kernel_initializer=tf.constant_initializer([
                [ 1/16, 1/8, 1/16 ],
                [ 1/8, 1/4, 1/8 ],
                [ 1/16, 1/8, 1/16 ],
            ]),
        )
    ])

    with tf.GradientTape() as tape:
        tape.watch(images)
        pred = model(images)
        loss = calculateLoss(labels, pred)

    gradient = tape.gradient(loss, images)
    
    pool_0 = tf.sign(gradient) * epsilon
    pool_1 = calculatePool(pool_0)
    pool_2 = calculatePool(pool_1)
    
    bounds = epsilon * .1
    images = images + tf.squeeze(pool_2)
    images = images + tf.random.uniform(
        shape=tf.shape(images),
        minval=-bounds,
        maxval=+bounds
    )

    return tf.clip_by_value(images, clip[0], clip[1])
