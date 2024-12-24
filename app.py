import tensorflow as tf
import os

# Define dataset directory paths
train_dir = "C:/Users/Mabelle/Desktop/DiseaseDiagnosis/chest-xray-pneumonia/chest_xray/train"
test_dir = "C:/Users/Mabelle/Desktop/DiseaseDiagnosis/chest-xray-pneumonia/chest_xray/test"
val_dir = "C:/Users/Mabelle/Desktop/DiseaseDiagnosis/chest-xray-pneumonia/chest_xray/val"

# Verify if the directories exist
print(f"Train Directory exists: {os.path.exists(train_dir)}")
print(f"Test Directory exists: {os.path.exists(test_dir)}")
print(f"Validation Directory exists: {os.path.exists(val_dir)}")

# Set image size for preprocessing
image_size = (150, 150)

# Load datasets from the directories
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=32,
    label_mode='binary',
    shuffle=True
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=32,
    label_mode='binary',
    shuffle=False
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=image_size,
    batch_size=32,
    label_mode='binary',
    shuffle=False
)

# Normalize the data
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Print a sample of the data
for images, labels in train_dataset.take(1):
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(150, 150, 3)),  # Rescale pixel values to [0,1]
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()  # Print the model summary

# Train the model
history = model.fit(
    train_dataset,  # Training dataset
    validation_data=val_dataset,  # Validation dataset
    epochs=10  # Number of training epochs
)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save('chest_xray_model.h5')

