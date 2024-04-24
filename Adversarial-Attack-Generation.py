def fgsm_attack(image, epsilon, gradient):
    """
    Generates adversarial image using FGSM.

    Args:
    - image (tensor): Original image.
    - epsilon (float): Perturbation amount.
    - gradient (tensor): Gradient of the loss with respect to the input image.

    Returns:
    - tensor: Perturbed image.
    """
    # Get the sign of the gradient
    signed_grad = tf.sign(gradient)
    # Create the perturbed image by adjusting each pixel of the input image
    adv_image = image + epsilon * signed_grad
    # Clip the perturbed image to [0,1] range
    adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image



def generate_adversarial_examples(model, X, y, epsilon=0.01):
    adv_images = []

    for image, label in zip(X, y):
        image_placeholder = tf.Variable(image.reshape((1, 152, 152, 3)))

        with tf.GradientTape() as tape:
            tape.watch(image_placeholder)
            prediction = model(image_placeholder)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=[label], y_pred=prediction, from_logits=True)

        gradient = tape.gradient(loss, image_placeholder)

        if gradient is None:
            print("Gradient is None for label:", label)
            continue

        adv_image = fgsm_attack(image_placeholder, epsilon, gradient)
        adv_images.append(adv_image.numpy().squeeze())

    return np.array(adv_images)
