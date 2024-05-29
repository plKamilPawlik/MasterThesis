import tensorflow as tf

def FastFoolAttack(
    model,
    images,
    labels,
    epsilon: float,
    upsilon: float,
    batch_size: int,
    batch_iter: int,
    clip=(0, 1),
):
    calculateLoss = tf.keras.losses.SparseCategoricalCrossentropy()

    # helper function
    def flatten(inputs):
        return tf.clip_by_value(inputs, clip[0], clip[1])

    # helper function
    def uniform(inputs):
        return tf.random.uniform(
            shape=tf.shape(inputs),
            minval=0.97,
            maxval=1.02,
        )

    with tf.GradientTape() as tape:
        tape.watch(images)
        pred = model(images)
        loss = calculateLoss(labels, pred)

    # calculate gradient and perturbations
    grad = tape.gradient(loss, images)
    sign = tf.sign(grad) * epsilon

    # calculate initial loss
    pred = model(flatten(images + sign))
    loss = calculateLoss(labels, pred)

    # initialize perturbations
    pert = tf.Variable(sign)

    for batch in range(0, images.shape[0], batch_size):
        batch_images = images[batch : batch + batch_size]
        batch_labels = labels[batch : batch + batch_size]

        # batch robust
        batch_pert = pert[batch : batch + batch_size]
        batch_pred = model(flatten(batch_images + batch_pert))
        batch_loss = calculateLoss(batch_labels, batch_pred)

        # batch energy
        batch_energy = (batch_pert.numpy() ** 2).sum()

        for biter in range(batch_iter):
            biter_pert = batch_pert * uniform(batch_pert)
            biter_pred = model(flatten(batch_images + biter_pert))
            biter_loss = calculateLoss(batch_labels, biter_pred)

            # batch energy
            biter_energy = (biter_pert.numpy() ** 2).sum()
            biter_robust = (tf.math.log(upsilon * batch_loss + 1) / upsilon)

            if (batch_energy > biter_energy) and (biter_loss > biter_robust):
                batch_energy = biter_energy
                batch_pert = biter_pert

        # assign for batch
        pert[batch : batch + batch_size].assign(batch_pert)

    # discredit positive energy
    pert = sign - tf.clip_by_value(sign - pert, -float('inf'), 0)

    return flatten(images + pert)
