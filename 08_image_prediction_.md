# Chapter 8: Image Prediction

Welcome back! In our previous chapters, we've journeyed through building our Convolutional Neural Network (CNN) ([Chapter 1: CNN Model Architecture](01_cnn_model_architecture_.md)), preparing and augmenting our data ([Chapter 2: Dataset Preparation](02_dataset_preparation_.md), [Chapter 3: Data Augmentation](03_data_augmentation_.md), [Chapter 4: Keras ImageDataGenerator](04_keras_imagedatagen_ator_.md)), training the model ([Chapter 5: Model Training](05_model_training_.md), [Chapter 6: Keras Callbacks](06_keras_callbacks_.md)), and evaluating how well it learned using test data ([Chapter 7: Model Evaluation and Metrics](07_model_evaluation_and_metrics_.md)).

Now that we have a trained and evaluated model, the ultimate goal is to use it for its intended purpose: classifying *new*, unseen brain MRI images. This is the step where our hard work translates into a practical application. This process is called **Image Prediction**.

## The Goal: Classifying a Single New Brain Scan

Imagine a doctor has a new brain MRI scan for a patient. They want to know if the scan shows a tumor, and if so, what type (Glioma, Meningioma, or Pituitary). Our trained CNN model is designed to help with this!

**Image Prediction** is the process of taking one of these new, individual images and feeding it into our trained model to get a classification result.

## What Happens During Prediction?

Think back to our student analogy. The student (our CNN) has studied extensively and passed a final exam (model evaluation). Now, we give the student a single, new flashcard they've never seen before and ask, "What is this?"

The student looks at the features of the image and, based on everything they've learned, makes an educated guess. In our CNN's case, its "guess" is a set of probabilities – how likely the image is to belong to each of the possible categories (Glioma, Meningioma, Pituitary, No Tumor).

The model will output something like:
*   Glioma: 0.05 (5% probability)
*   Meningioma: 0.01 (1% probability)
*   Pituitary: 0.93 (93% probability)
*   No Tumor: 0.01 (1% probability)

Since the "Pituitary Tumor" category has the highest probability (93%), the model's prediction for this image is "Pituitary Tumor".

## Steps to Make a Prediction

To classify a new image using our trained model, we need to follow these steps:

1.  **Load the Trained Model:** We need to load the model we saved after training (which might have happened using a callback like `ModelCheckpoint` or explicitly after `model.fit`).
2.  **Load the New Image:** Read the new brain scan image file into our program.
3.  **Preprocess the Image:** Just like we did for the training data ([Chapter 2: Dataset Preparation](02_dataset_preparation_.md)), the new image must be resized to the exact dimensions our CNN expects (150x150 pixels in our case) and the pixel values scaled (from 0-255 to 0-1).
4.  **Reshape for the Model:** Our CNN was trained on *batches* of images. Even for a single image prediction, the model still expects the input data to have a "batch dimension". We need to add this.
5.  **Run the Prediction:** Feed the prepared and reshaped image data into the loaded model's `predict()` method.
6.  **Interpret the Output:** The model outputs probabilities for each class. We find the class with the highest probability and map that back to the actual class name.

Let's look at how this translates into code.

## Implementing Image Prediction in Code

We'll demonstrate the prediction process using a hypothetical single image file.

First, we need to load the trained model. Assuming you saved your model to a file named `brain_tumor_model.h5` or a directory like `brain_tumor_model_dir`:

```python
import tensorflow as tf

# Load the saved model
try:
    # If saved as an HDF5 file
    # model = tf.keras.models.load_model('brain_tumor_model.h5')

    # If saved in the TensorFlow SavedModel format (e.g., using model.save('my_model_dir'))
    model = tf.keras.models.load_model('D:/PBS/vggmodel') # Replace with your actual model path

    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file/directory exists at the specified path.")
    # You might want to exit or handle the error appropriately here
```
*   **Explanation:** We use `tf.keras.models.load_model()` to load the entire model architecture and its learned weights from where it was saved. The path `'D:/PBS/vggmodel'` comes from the project's notebook saving step.

Next, we need to load and preprocess the new image. We'll need the `cv2` library for image reading and resizing, and `numpy` for array manipulation, just like in [Chapter 2: Dataset Preparation](02_dataset_preparation_.md).

```python
import cv2
import numpy as np
import os

# Define the target image size (must match the model's input size)
IMAGE_SIZE = 150

# Define the original labels list (from Chapter 2)
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
# Note: The project's notebook uses 3 classes in the model/evaluation, but the loading code
# and the original dataset structure imply 4. We'll use 4 here for generality,
# assuming 'no_tumor' is included in the final model. If the model predicts 3,
# the last Dense layer should have 3 units, and the labels list should match.
# Let's adjust the labels list here based on the notebook's final output (3 classes)
labels = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']


# Path to a new image file you want to classify
# Replace with a path to an actual test image or a new image
new_image_path = 'D:/PBS/labelled Dataset/Testing/pituitary_tumor/Tp_0012.jpg' # Example path

# Check if the image file exists
if not os.path.exists(new_image_path):
    print(f"Error: Image file not found at {new_image_path}")
    # Handle error or exit
else:
    # Load the image
    img = cv2.imread(new_image_path)

    # Check if image loading was successful
    if img is None:
        print(f"Error: Could not load image from {new_image_path}")
        # Handle error or exit
    else:
        # Resize the image to the target size
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Scale pixel values from 0-255 to 0-1 (as done during training)
        # Assuming the model expected data in the 0-1 range (e.g., if rescale=1./255 was used)
        img = img / 255.0

        # Convert the image to a NumPy array (it should already be, but good practice)
        img_array = np.array(img)

        # Add a batch dimension: reshape from (height, width, channels) to (1, height, width, channels)
        img_for_prediction = np.expand_dims(img_array, axis=0)

        print(f"Image loaded and preprocessed: shape {img_for_prediction.shape}")

```
*   **Explanation:** We load the image using `cv2.imread`, resize it to `IMAGE_SIZE` (150x150), and then scale the pixel values by dividing by 255.0. The critical step for prediction is `np.expand_dims(img_array, axis=0)`. Our model expects input in the shape `(batch_size, height, width, channels)`. Since we have only one image, the `batch_size` is 1. `np.expand_dims` adds this extra dimension at the beginning (axis 0).

Now that the image is in the correct format, we can use the `model.predict()` method.

```python
# Check if the image was successfully loaded before predicting
if 'img_for_prediction' in locals():
    # Get the model's predictions
    predictions = model.predict(img_for_prediction)

    print(f"Raw model output (probabilities): {predictions}")

    # The output is an array of probabilities, e.g., [[0.05, 0.93, 0.01, 0.01]]
    # Find the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Map the index back to the class label string
    predicted_class_label = labels[predicted_class_index]

    # Get the probability of the predicted class
    predicted_probability = predictions[0, predicted_class_index]

    print(f"Predicted class index: {predicted_class_index}")
    print(f"Predicted class label: {predicted_class_label}")
    print(f"Predicted probability: {predicted_probability:.4f}")

    # Interpret the result in a friendly way
    if predicted_class_label == 'no_tumor':
         print("\nThe model predicts: No tumor detected.")
    else:
         print(f"\nThe model predicts: {predicted_class_label} detected.")

```
*   **Explanation:** `model.predict(img_for_prediction)` takes the prepared image (now a batch of size 1) and runs it through the CNN. The output `predictions` is a NumPy array of shape `(1, number_of_classes)`, where each value is the probability for a class. `np.argmax(predictions, axis=1)` finds the index of the maximum probability in each row of the predictions array (since there's only one row, it returns a single index). We use `[0]` to get that single index value. Finally, we use this index to look up the corresponding string label from our `labels` list.

This process allows us to classify any new brain MRI image using our trained model.

## How Image Prediction Works (High-Level Flow)

Let's visualize the prediction process for a single image:

```mermaid
sequenceDiagram
    participant New Image File
    participant Preprocessing Code
    participant Prepared Image Data (1, 150, 150, 3)
    participant Trained CNN Model
    participant Model Output (Probabilities)
    participant Prediction Logic
    participant Final Class Label

    New Image File->>Preprocessing Code: Load & Resize
    Preprocessing Code->>Prepared Image Data (1, 150, 150, 3): Scale Pixels, Add Batch Dim.
    Prepared Image Data (1, 150, 150, 3)->>Trained CNN Model: Input for Prediction
    Trained CNN Model->>Trained CNN Model: Process through layers (no training!)
    Trained CNN Model-->>Model Output (Probabilities): Output Scores (e.g., [0.05, 0.01, 0.93, 0.01])
    Model Output (Probabilities)->>Prediction Logic: Find max probability index
    Prediction Logic->>Final Class Label: Map index to label string ("Pituitary Tumor")
    Final Class Label-->>User: Display Result
```
*   **Explanation:** The new image is loaded and processed to match the format the CNN understands (resized, scaled, batch dimension added). This prepared data is passed to the trained model's `predict` function. The model performs a forward pass through its layers, calculating output probabilities. The highest probability determines the predicted class, which is then presented to the user. Notice there's no loss calculation or weight updates – the model is only making a prediction based on what it has already learned.

The `img_pred` function included in the project's notebook (`compiled_layered_model.ipynb`) or `classification.py` encapsulates these steps, allowing a user to upload an image and get a prediction directly. It also includes the logic to check if the prediction matches the actual label for test images, providing visual feedback (the red/green labels in the plot from [Chapter 7](07_model_evaluation_and_metrics_.md) are based on this comparison).

## Why These Steps?

*   **Loading the Model:** A model is useless if you can't load the result of training! Saving and loading allow us to deploy the model without retraining.
*   **Preprocessing:** The model learned to recognize patterns in images of a specific size and value range. New images must match this format for the model to understand them correctly.
*   **Reshaping:** CNNs process data in batches for efficiency, even if the batch size is 1 during prediction. Adding the batch dimension provides the data shape the model's input layer expects.
*   **`model.predict()`:** This is the Keras method specifically designed for inference (making predictions) using a trained model. It performs the forward pass without calculating gradients or updating weights.
*   **Interpreting Output:** The raw output of the `softmax` layer is a vector of probabilities. We need `np.argmax` to easily get the most likely class and our original `labels` list to translate the numerical index into a meaningful category name for a human user.

## Conclusion

In this chapter, we learned how to use our trained CNN model to make predictions on new, individual brain MRI images. We walked through the essential steps of loading the model, preparing a new image, running the prediction, and interpreting the probabilistic output to arrive at a final classification. This prediction process is the culmination of our efforts in building and training the model and represents the practical application of deep learning for brain tumor classification.

With the ability to predict, our project can now be used to potentially assist in the classification of real-world brain scans.
