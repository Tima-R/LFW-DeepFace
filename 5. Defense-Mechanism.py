# Adversarial Training
def adversarial_training(model, X_train, y_train, X_val, y_val, epsilon, epochs=5, batch_size=32):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Shuffle and batch the training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for start_idx in range(0, X_train.shape[0] - batch_size + 1, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # Vectorized generation of adversarial examples
            X_batch_tensor = tf.convert_to_tensor(X_batch)
            with tf.GradientTape() as tape:
              tape.watch(X_batch_tensor)
              predictions = model(X_batch_tensor)
              loss = tf.keras.losses.categorical_crossentropy(y_true=y_batch, y_pred=predictions)
            gradients = tape.gradient(loss, X_batch_tensor)
            adv_images_batch = fgsm_attack(X_batch_tensor, epsilon, gradients)


            # Train model on the adversarial batch
            model.train_on_batch(adv_images_batch, y_batch)

        # Evaluate the model on validation set
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Calling the function
adversarial_training(modified_model, X_train, y_train_cat, X_test, y_test_cat, epsilon=0.01, epochs=5, batch_size=32)
