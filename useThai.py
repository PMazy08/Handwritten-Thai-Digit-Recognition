import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('thai_digits_model2.h5')

# Check if the model loaded successfully
if model:
    image_number = 1
    while os.path.isfile(f"thaidigits/thaidigit{image_number}.png"):
        try:
            # Read and preprocess the image
            img = cv2.imread(f"thaidigits/thaidigit{image_number}.png", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))  # Resize the image
            img = np.invert(img)  # Invert the colors
            img = img.astype('float32') / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Perform prediction
            prediction = model.predict(img)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            print(f"This digit is probably a {predicted_digit} with confidence {confidence:.2f}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except Exception as e:
            print('Error:', e)
        finally:
            image_number += 1
else:
    print("Failed to load the model.")