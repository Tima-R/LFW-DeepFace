from tensorflow.keras.models import Model


# Convert labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# Load the DeepFace model without the final layer
base_model = DeepFace.build_model("DeepFace")


# Add the final classification layers
x = base_model.output
x = Dense(7, activation='softmax')(x)

# Define the new model
modified_model = Model(inputs=base_model.input, outputs=x)


# Model compilation
opt = Adam(learning_rate=0.0005)
modified_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = modified_model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=5, batch_size=32)

print("Training completed!")
