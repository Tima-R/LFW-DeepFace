# Load the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data
y = lfw_people.target

# Reshape the flattened images
X = X.reshape((-1, 50, 37))

# Resize the images to (152, 152)
X_resized = np.array([cv2.resize(img, (152, 152)) for img in X])

# Convert grayscale images to pseudo RGB
X_rgb = np.repeat(X_resized[..., np.newaxis], 3, axis=-1)


print(X_train.shape)  # This should output (1030, 152, 152, 3)

# Convert grayscale images to pseudo RGB
X_rgb = np.repeat(X_resized[..., np.newaxis], 3, axis=-1)

# Print shapes for debugging
print("Shape of X_resized:", X_resized.shape)
print("Shape of X_rgb:", X_rgb.shape)

# Normalize the dataset
X_rgb = X_rgb / 255.0

# Splitting the training data into 80% training and 20% validation
X_train, X_test, y_train, y_test = train_test_split(X_rgb, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
