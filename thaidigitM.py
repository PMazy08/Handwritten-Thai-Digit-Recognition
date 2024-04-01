import tensorflow as tf
import os
import cv2
import numpy as np

# Load ThaiDigits dataset
def load_and_process_data(data_dir):
    images = []
    labels = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            label = int(folder_name)  
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
                img = cv2.resize(img, (28, 28)) 
                img = np.invert(img) 
                img = img.astype('float32') / 255.0  
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Define paths to ThaiDigits dataset
train_data_dir = 'dataset/train'
test_data_dir = 'dataset/test'

# Load and preprocess the data
x_train, y_train = load_and_process_data(train_data_dir)
x_test, y_test = load_and_process_data(test_data_dir)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test))

# Save the model
model.save('thai_digits_model99.h5')
