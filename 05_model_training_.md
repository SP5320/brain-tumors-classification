# Chapter 5: Model Training

Welcome back! In our previous chapters, we've laid the groundwork for our brain tumor classification project. We designed the core "brain" of our system, the Convolutional Neural Network (CNN) ([Chapter 1: CNN Model Architecture](01_cnn_model_architecture_.md)). We also prepared our image data, organizing it and transforming it into a format suitable for the CNN ([Chapter 2: Dataset Preparation](02_dataset_preparation_.md)). We even learned about Data Augmentation ([Chapter 3: Data Augmentation](03_data_augmentation_.md)) and how Keras's `ImageDataGenerator` ([Chapter 4: Keras ImageDataGenerator](04_keras_imagedatagenerator_.md)) can help us create more diverse training examples to make our model more robust.

Now, with a well-structured model and carefully prepared data (including augmented images), it's time for the most exciting step: **Model Training**!

## What is Model Training?

Think of model training like teaching a student to identify different types of brain tumors by showing them thousands of labeled examples (our prepared dataset). The student (our CNN model) starts with no knowledge. We show them a picture and ask, "Is this a Glioma, Meningioma, Pituitary, or No Tumor?"

Initially, the student will guess randomly, and they'll probably be wrong most of the time. We (the training process) tell them the correct answer and **show them how wrong their guess was**. Based on how wrong they were, they **adjust their internal understanding** (like studying harder on areas they got wrong) so they can make better guesses next time.

This process of showing examples, guessing, getting feedback, and adjusting is repeated thousands and thousands of times. Gradually, the student gets better and better at recognizing the visual patterns associated with each tumor type.

In deep learning terms:

*   **The Student:** Our CNN model with its layers and millions of internal parameters (numbers called 'weights' and 'biases').
*   **Showing Examples:** Feeding batches of images and their correct labels from our prepared training data (`X_train`, `y_train`).
*   **Guessing:** The CNN processing the image through its layers and producing a prediction (e.g., a probability for each tumor type).
*   **Feedback (How Wrong):** This is measured by a **Loss Function**. It calculates a single number that tells us how far the model's prediction was from the actual correct label. A high loss means the model was very wrong, a low loss means it was close or correct.
*   **Adjusting Internal Understanding:** This is handled by an **Optimizer**. Based on the loss function's feedback, the optimizer figures out how to slightly change the model's weights and biases to reduce the loss for that specific batch of data.

The entire training process is about finding the *best* set of weights and biases for the CNN so that, when shown a new brain scan image, it can predict the correct tumor type (or no tumor) with high accuracy.

## Core Components of Training in Keras

Before we start the repetitive learning process, we need to tell our Keras model *how* to learn. We do this using the `model.compile()` method. This method configures the model for training by specifying:

*   **The Optimizer:** The algorithm that will adjust the model's weights. A common and effective one is called 'Adam'.
*   **The Loss Function:** The way the model measures how wrong its predictions are. For multi-class classification (like predicting one of four tumor types), the 'categorical_crossentropy' loss is standard when using one-hot encoded labels ([Chapter 2: Dataset Preparation](02_dataset_preparation_.md)).
*   **The Metrics:** How we want to evaluate the model's performance during training. 'accuracy' is the most common metric for classification tasks, measuring the percentage of correct predictions.

Here's how this looks in the project code (`compiled_layered_model.ipynb` or `classification.py`):

```python
model = Sequential() # Assuming your model architecture is defined as in Chapter 1
# ... add layers ...
model.add(Dense(4, activation='softmax')) # Example for 4 classes

# Compile the model
model.compile(
    loss='categorical_crossentropy',  # The loss function for multi-class classification
    optimizer='Adam',                # The optimization algorithm
    metrics=['accuracy']             # How to evaluate performance
)
```
*   **Explanation:** We call `model.compile()` on our `Sequential` model. We tell it to use `categorical_crossentropy` loss (because our `softmax` output layer predicts probabilities for each class, and our labels are one-hot encoded), the `Adam` optimizer to handle weight updates, and to report `accuracy` as it trains.

## The Training Loop: `model.fit()`

Once the model is compiled, the actual training begins using the `model.fit()` method. This method takes the training data and validation data and runs the training process for a specified number of epochs.

Key parameters for `model.fit()` when using NumPy arrays:

| Parameter         | What it is                                                                 | Analogy                       |
| :---------------- | :------------------------------------------------------------------------- | :---------------------------- |
| `x` (or `X_train`)| The training data (the image arrays).                                      | The stack of flashcards.      |
| `y` (or `y_train`)| The training labels (the one-hot encoded labels for the images).           | The answers on the back.      |
| `epochs`          | The number of times to iterate over the entire training dataset.             | Number of full study sessions. |
| `batch_size`      | The number of samples (images and labels) to process in each training step. | Size of one small stack.      |
| `validation_data` | Data to evaluate the model on *after* each epoch (`X_test`, `y_test` pair).| A quiz after each study session. |
| `verbose`         | How much information to print during training (1 for progress bar).        | How detailed the study log is. |

The project code uses NumPy arrays that were prepared and pre-augmented in the data loading step ([Chapter 2](02_dataset_preparation_.md) and the augmentation part of the code in [Chapter 3](03_data_augmentation_.md)). Let's look at the `model.fit()` call from the project code:

```python
# Assuming X_train, y_train, X_test, y_test are NumPy arrays
# prepared and encoded as described in Chapters 2 & 4.

history = model.fit(
    X_train, y_train,             # Training data and labels
    validation_split=0.1,         # Use 10% of X_train for validation *during* training
    epochs=20,                    # Run for 20 full passes through the data
    verbose=1,                    # Show training progress
    batch_size=32,                # Process 32 images at a time
    validation_data=(X_test, y_test), # Evaluate on the separate test set after each epoch
    callbacks=[tensorboard, reduce_lr] # Use callbacks (covered in Chapter 6)
)
```
*   **Explanation:** This code initiates the training. The model will iterate through the `X_train` data (in batches of 32 images) 20 times (`epochs=20`). After each epoch, it will evaluate its performance on the separate `X_test` and `y_test` data (`validation_data`). The `validation_split=0.1` argument here is *redundant* because `validation_data` is also provided. Keras will prioritize `validation_data` if both are present. The `verbose=1` makes Keras print a progress bar and metrics for each epoch. The `callbacks` are extra tools we'll discuss in the next chapter.

This process continues for 20 epochs, with the optimizer (`Adam`) constantly adjusting the model's internal weights based on the loss (`categorical_crossentropy`) calculated after processing each batch. The `accuracy` metric is monitored to see how well the model is learning to classify correctly.

## How Model Training Works (High-Level Flow)

Let's visualize the iterative learning process that happens inside `model.fit()`:

```mermaid
sequenceDiagram
    participant Training Loop (model.fit)
    participant Training Data (X_train, y_train)
    participant CNN Model
    participant Loss Function
    participant Optimizer

    Training Loop (model.fit)->>Training Loop (model.fit): Start Epoch 1
    Training Loop (model.fit)->>Training Data (X_train, y_train): Get a batch of images and labels
    Training Data (X_train, y_train)-->>CNN Model: Batch of Images (X)
    CNN Model->>CNN Model: Process images through layers
    CNN Model-->>Loss Function: Model Predictions (ŷ)
    Loss Function->>Training Data (X_train, y_train): Actual Labels (y) for the batch
    Loss Function->>Loss Function: Calculate Loss (e.g., categorical_crossentropy)
    Loss Function-->>Optimizer: Loss Value
    Optimizer->>Optimizer: Calculate how to adjust weights (using gradients)
    Optimizer-->>CNN Model: Apply Weight/Bias Updates
    Note over Training Loop (model.fit): Repeat for all batches in the epoch
    Training Loop (model.fit)->>Training Loop (model.fit): End Epoch 1, Evaluate on Validation Data
    Note over Training Loop (model.fit): Repeat for all specified epochs
```
*   **Explanation:** The training loop manages the epochs. Within each epoch, it grabs batches of training data, feeds them to the CNN, gets predictions, calculates the loss by comparing predictions to actual labels, and then uses the optimizer to update the model's parameters to reduce that loss. This cycle repeats for every batch in the epoch, and then for every epoch, providing the model with thousands of opportunities to learn and refine its ability to classify images.

## Visualizing Training Progress

The `history` object returned by `model.fit()` contains a record of the loss and metric values (like accuracy) at the end of each epoch for both the training data and the validation data. We can plot these values over epochs to see how well the model is learning and if it's overfitting.

Looking at the project's notebook (`compiled_layered_model.ipynb`), you'll see plots like these generated:

```python
# Assuming 'history' is the object returned by model.fit()
# This code plots the accuracy over epochs
plt.figure(figsize=(18,12))
plt.plot(history.history['val_accuracy'], marker = 'o') # Validation accuracy
plt.plot(history.history['accuracy'], marker = 'o')     # Training accuracy
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Validation Accuracy','Training Accuracy'])
plt.show()

# This code plots the loss over epochs
plt.figure(figsize=(18,12))
plt.plot(history.history['val_loss'], marker = 'o')     # Validation loss
plt.plot(history.history['loss'], marker = 'o')         # Training loss
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Validation Loss','Training Loss'])
plt.show()
```
*   **Explanation:** These plots are crucial for monitoring training. Ideally, you want to see both training and validation accuracy increase over epochs, and both training and validation loss decrease.
    *   If training accuracy is much higher than validation accuracy, or training loss is much lower than validation loss, the model might be **overfitting** (memorizing the training data but not generalizing well).
    *   If both curves flatten out and aren't improving, the model might have **converged** (learned as much as it can from the current setup) or it might be **underfitting** (not powerful enough or hasn't trained long enough).

The plots in the project's notebook show that the model's validation accuracy and loss improve significantly in the first few epochs and then plateau, indicating it has learned effectively within 20 epochs.

## Conclusion

In this chapter, we took the crucial step of training our CNN model. We learned about the core concepts behind the learning process: the loss function measuring error, the optimizer adjusting weights, and metrics like accuracy tracking performance. We saw how Keras uses the `model.compile()` method to set up the training process and the `model.fit()` method to execute the training loop using our prepared data. We also learned how to visualize the training progress to understand how well our model is learning.

While training is the core learning step, there are additional tools in Keras called Callbacks that can help us make the training process even better. We'll explore these in the next chapter.

[Keras Callbacks](06_keras_callbacks_.md)
