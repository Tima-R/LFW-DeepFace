## Baseline Performance

# Evaluate the model's performance on the original test set
baseline_loss, baseline_accuracy = modified_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Baseline Performance - Loss: {baseline_loss:.4f}, Accuracy: {baseline_accuracy:.4f}")



## Under Attack

def generate_adversarial_examples(model, X_data, epsilon):
    adversarial_samples = []
    for image in X_data:
        image_to_perturb = tf.expand_dims(image, 0)
        with tf.GradientTape() as tape:
            tape.watch(image_to_perturb)
            prediction = model(image_to_perturb)
            true_label = tf.argmax(prediction, axis=1)
            true_label_one_hot = tf.one_hot(true_label, depth=7)
            loss = tf.keras.losses.categorical_crossentropy(y_true=true_label_one_hot, y_pred=prediction)
        gradient = tape.gradient(loss, image_to_perturb)
        adv_image = fgsm_attack(image_to_perturb, epsilon, gradient)
        adversarial_samples.append(adv_image.numpy().squeeze())
    return np.array(adversarial_samples)

# Generate adversarial examples for test set
X_test_adv = generate_adversarial_examples(modified_model, X_test, epsilon=0.01)

# Evaluate model performance under attack
attack_evaluation = modified_model.evaluate(X_test_adv, y_test_cat, verbose=0)
print(f"Performance Under FGSM Attack - Loss: {attack_evaluation[0]:.4f}, Accuracy: {attack_evaluation[1]:.4f}")


## Defense Performance

# Evaluate the modified model performance under FGSM attack
defense_evaluation = modified_model.evaluate(X_test_adv, y_test_cat, verbose=0)
print(f"Performance with Defense Under FGSM Attack - Loss: {defense_evaluation[0]:.4f}, Accuracy: {defense_evaluation[1]:.4f}")


# Print accuracy comparisons
print(f"Baseline Accuracy: {baseline_evaluation[1]:.4f}")
print(f"Accuracy Under FGSM Attack: {attack_evaluation[1]:.4f}")
print(f"Accuracy with Defense Under FGSM Attack: {defense_evaluation[1]:.4f}")


# Assuming `model` is your original model and `X_test`, `y_test` are test data and labels.
baseline_evaluation = modified_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Baseline Performance - Loss: {baseline_evaluation[0]:.4f}, Accuracy: {baseline_evaluation[1]:.4f}")
