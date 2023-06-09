import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert class labels to one-hot encoded vectors
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Method 1: Conv Net + Max Pooling (hand tuned)
model1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

# Method 2: Conv Net + Max Pooling (Snoek et al., 2012)
model2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

# Method 3: Conv Net + Max Pooling + Dropout in Fully Connected Layers
model3 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Method 4: Conv Net + Max Pooling + Dropout in All Layers
model4 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Method 5: Conv Net + Maxout (Goodfellow et al., 2013)
model5 = Sequential([
    Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), padding='same'),
    Conv2D(512, (3, 3), padding='same'),
    Flatten(),
    Dense(512, activation='maxout'),
    Dense(10, activation='softmax')
])

# Compile the models
models = [model1, model2, model3, model4, model5]
for model in models:
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the models
history = []
for model in models:
    h = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test), verbose=0)
    history.append(h)

# Evaluate the models
test_loss = []
test_acc = []
for model in models:
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    test_loss.append(loss)
    test_acc.append(acc)

# Plot the comparison of test accuracy for each method
plt.bar(range(5), test_acc)
plt.xticks(range(5), ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5'])
plt.xlabel('Methods')
plt.ylabel('Test Accuracy')
plt.title('Comparison of Test Accuracy for Different Methods')
plt.show()

# Plot the loss vs. epoch curves for each method
plt.figure(figsize=(10, 6))
for i, h in enumerate(history):
    plt.plot(h.history['loss'], label=f'Method {i+1}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs for Different Methods')
plt.legend()
plt.show()
