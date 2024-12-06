

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
# Define the source directory containing your dataset
source_dir = "C:\\Users\\Hp\\Downloads\\New folder\\flowers"

# Convert the source directory path to a pathlib.Path object
data_dir = pathlib.Path(source_dir)
# Define image dimensions
img_height = 180
img_width = 180

# Load and preprocess the dataset
batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
# Extract class names
class_names = train_ds.class_names
num_classes = len(class_names)
class_names
# Define the ResNet50 base model
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = False
# Add custom classification head
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = tf.keras.applications.resnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam()

# Train the model
epochs = 20

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Define the training step function inside the loop
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # Forward pass passing the input images through the model to get predictions.
        predictions = model(images)
        # Compute loss between labels and prediction
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        loss = tf.reduce_mean(loss)
    # the model's weights are adjusted in a way that reduces the loss improving over gradient decent
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_train_losses = []
    epoch_train_accuracies = []
    for images, labels in train_ds:
        # Perform training step
        loss = train_step(images, labels)
        epoch_train_losses.append(loss)
        predictions = model(images)
        acc = tf.keras.metrics.sparse_categorical_accuracy(labels, predictions)
        epoch_train_accuracies.extend(acc)
    # Compute average training loss and accuracy for the epoch
    avg_train_loss = np.mean(epoch_train_losses)
    avg_train_accuracy = np.mean(epoch_train_accuracies)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)
    print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss}, Average Training Accuracy: {avg_train_accuracy}")
     # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(val_ds)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Validation Accuracy: {val_accuracy}, Validation Loss: {val_loss}")

print("Training complete.")
# Visualize training and validation metrics
epochs_range = range(1, epochs+1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
# Save the model
try:
    model.save('/content/drive/MyDrive/Project/modelRes.h5')
    print("Model saved successfully.")
except Exception as e:
    print("Error saving the model:", e)
    # Save the model using tf.keras.models.save_model()
tf.keras.models.save_model(model, '/content/drive/MyDrive/Project/modelres.h5')