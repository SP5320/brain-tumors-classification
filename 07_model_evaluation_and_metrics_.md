# Chapter 7: Model Evaluation and Metrics

Welcome back! In the last chapter, [Keras Callbacks](06_keras_callbacks_.md), we learned how Keras callbacks like `ReduceLROnPlateau` can help improve and automate the training process of our Convolutional Neural Network (CNN) ([Chapter 5: Model Training](05_model_training_.md)).

We've built our model and trained it on the training data (including augmented images from [Chapter 3: Data Augmentation](03_data_augmentation_.md) and [Chapter 4: Keras ImageDataGenerator](04_keras_imagedatagenerator_.md), prepared in [Chapter 2: Dataset Preparation](02_dataset_preparation_.md)). But how do we know how *good* our model actually is at classifying brain scans?

Training performance (like the training accuracy we saw in [Chapter 5: Model Training](05_model_training_.md)) is helpful, but it tells us how well the model did on the data it *practiced* on. The real test is how it performs on brain scans it has **never seen before**.

## The Problem: How Do We Know Our Model Works on New Images?

Imagine a student studying for an exam. If you give them the exact same questions during the exam that they practiced with, they'll likely get a perfect score. But this doesn't tell you if they truly *understood* the subject or just memorized the answers.

Similarly, a machine learning model might perform perfectly on the training data but fail miserably when given new, unseen examples. This is the problem of **overfitting** that we discussed in [Chapter 3: Data Augmentation](03_data_augmentation_.md).

To truly assess our model's understanding and its ability to classify *new* brain tumors, we need to test it on a separate dataset – our **testing dataset** (`X_test`, `y_test`) that we set aside during [Chapter 2: Dataset Preparation](02_dataset_preparation_.md).

## Model Evaluation: Giving the Model an Exam

Model evaluation is the process of feeding the testing dataset to our trained model and measuring how accurately it predicts the correct class (tumor type or no tumor) for each image. This gives us an unbiased assessment of the model's performance.

We don't train the model on the testing data; we only use it to calculate performance metrics *after* training is complete.

## Key Metrics for Classification

When evaluating a classification model, we don't just look at one number. Different metrics give us different insights into where the model is performing well and where it might be struggling. Here are the key metrics we'll use:

1.  **Overall Accuracy:**
    *   **What it is:** The simplest metric. It's the total percentage of correct predictions out of all the predictions made on the testing dataset.
    *   **Analogy:** The overall score on the final exam (e.g., 85%).
    *   **Usefulness:** Gives a quick general idea of performance.
    *   **Limitation:** Can be misleading if your classes are imbalanced (e.g., 90% 'No Tumor' images; a model that always predicts 'No Tumor' would have 90% accuracy but be useless). Our data augmentation ([Chapter 3](03_data_augmentation_.md)) helps with class balance.

2.  **Classification Report:**
    *   **What it is:** A detailed breakdown of performance metrics for *each individual class*. It typically includes Precision, Recall, and F1-score.
    *   **Analogy:** A report showing your score broken down by subject (e.g., Math, Science, History) on the exam.
    *   **Usefulness:** Helps identify which specific tumor types the model is good at classifying and which ones it struggles with.

    Let's briefly define the metrics within the report:
    *   **Precision:** Out of all the times the model *predicted* a specific class (e.g., 'Glioma'), how many times was it *actually* that class? (Helps understand false positives).
        *   *Analogy:* If you predicted 'Glioma' 10 times, and 8 of those were correct, your precision for 'Glioma' is 80%.
    *   **Recall (Sensitivity):** Out of all the images that *actually* belong to a specific class (e.g., all true 'Glioma' images), how many did the model correctly *identify* as that class? (Helps understand false negatives).
        *   *Analogy:* If there were 10 true 'Glioma' images, and your model correctly found 7 of them, your recall for 'Glioma' is 70%.
    *   **F1-Score:** A single score that balances both Precision and Recall. It's often a good overall measure for a single class.

3.  **Confusion Matrix:**
    *   **What it is:** A table that visually summarizes the performance of a classification model. Each row represents the *actual* class of the test images, and each column represents the *predicted* class by the model.
    *   **Analogy:** A table showing where each student's answers went wrong - how many times did a student answer "Cat" when the picture was actually a "Dog"?
    *   **Usefulness:** Shows exactly *where* the misclassifications are happening. You can see, for example, how many 'Glioma' tumors were misclassified as 'Meningioma', or how many 'No Tumor' scans were incorrectly labeled as a tumor type.

    Here's a simplified structure:

    |              | Predicted: Class 1 | Predicted: Class 2 | Predicted: Class 3 |
    | :----------- | :----------------- | :----------------- | :----------------- |
    | **Actual: Class 1** | Correct (True Positives) | Incorrect (False Positives for C2) | Incorrect (False Positives for C3) |
    | **Actual: Class 2** | Incorrect (False Negatives for C2) | Correct (True Positives) | Incorrect (False Positives for C3) |
    | **Actual: Class 3** | Incorrect (False Negatives for C3) | Incorrect (False Negatives for C3) | Correct (True Positives) |

    The numbers on the diagonal (top-left to bottom-right) show the counts of correctly classified images for each class. The numbers off the diagonal show the errors.

## How to Perform Evaluation in Keras/Scikit-learn

Based on the project code ([`compiled_layered_model.ipynb`](compiled_layered_model.ipynb) or [`classification.py`](classification.py)), we can perform these evaluations using Keras for overall accuracy and scikit-learn (`sklearn`) for the detailed report and confusion matrix.

We will use the `X_test` and `y_test` arrays that were created during [Chapter 2: Dataset Preparation](02_dataset_preparation_.md). Remember that `y_test` here refers to the one-hot encoded labels.

### 1. Overall Accuracy with `model.evaluate()`

After training (`model.fit()`), we can call `model.evaluate()` on the test data.

```python
# Assuming 'model' is your trained CNN model
# Assuming X_test and y_test are your NumPy arrays of test images and one-hot encoded labels

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
```
*   **Explanation:** `model.evaluate()` takes the test features (`X_test`) and the true test labels (`y_test`). It feeds the test data through the model, calculates the loss using the loss function specified during `model.compile()` ([Chapter 5: Model Training](05_model_training_.md)), and also calculates the metrics (like accuracy) specified during `model.compile()`. It returns these values. `verbose=1` makes it print a progress bar. The output will show the loss and accuracy achieved on the completely unseen test data.

Looking at the project's notebook output:
```
Epoch 20/20
9/9 [==============================] - 1s 87ms/step - loss: 0.0594 - accuracy: 0.9754 - val_loss: 0.3531 - val_accuracy: 0.8996 - lr: 7.2900e-07
```
The final `val_accuracy` (validation accuracy) shown by `model.fit()` on `validation_data=(X_test, y_test)` is the accuracy on the test set after the final epoch. In this specific run, it was `0.8996`, or about 90%. The `model.evaluate` call after training would confirm this final accuracy on the test set.

### 2. Classification Report and Confusion Matrix with Scikit-learn

To get the classification report and confusion matrix, we first need the model's predictions for *each individual test image*. Keras `model.predict()` gives us these predictions.

```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # A library for making nice plots, including heatmaps

# Assuming 'model' is your trained CNN model
# Assuming X_test and y_test are your NumPy arrays of test images and one-hot encoded labels
# Assuming 'labels' is your original list of class names (e.g., ['glioma_tumor', 'no_tumor', ...])

# Get the model's predictions for the test data
y_pred_probabilities = model.predict(X_test)

# The output of model.predict is probabilities for each class.
# We need to convert these probabilities to the predicted class index (0, 1, 2, ...)
# np.argmax finds the index of the highest probability for each prediction.
y_pred_classes = np.argmax(y_pred_probabilities, axis=1)

# The true test labels y_test are one-hot encoded.
# We need to convert them back to the class index format to compare with y_pred_classes.
y_true_classes = np.argmax(y_test, axis=1)

# Now we can use scikit-learn to generate the report and matrix

# --- Classification Report ---
print("--- Classification Report ---")
# The classification_report function needs the true class indices and predicted class indices
# We also provide the target_names (the original labels list) for clarity
print(classification_report(y_true_classes, y_pred_classes, target_names=labels))

# --- Confusion Matrix ---
print("\n--- Confusion Matrix ---")
# The confusion_matrix function needs the true class indices and predicted class indices
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix as a heatmap for better visualization
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# The mlxtend library can also plot a confusion matrix with percentages
# from mlxtend.plotting import plot_confusion_matrix
# fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(7.5, 7.5), class_names=labels, show_normed=True);
# plt.show()

```
*   **Explanation:**
    *   `model.predict(X_test)`: Runs the test images through the trained model and returns an array where each row is a list of probabilities (one probability for each class) for the corresponding image.
    *   `np.argmax(..., axis=1)`: This NumPy function finds the *index* of the maximum value in each row (axis=1) of the probability array. Since our `softmax` output layer gives the highest probability to the predicted class, `np.argmax` effectively tells us which class the model thinks each image belongs to. We do this for both the model's predictions (`y_pred_classes`) and the true one-hot encoded labels (`y_true_classes`) to get them into a comparable format (e.g., `[0, 1, 0, 2, ...]` instead of `[[1,0,0],[0,1,0],[1,0,0],[0,0,1], ...]`).
    *   `classification_report()`: Takes the actual class labels and the predicted class labels and generates the text report with precision, recall, and f1-score for each class.
    *   `confusion_matrix()`: Creates the confusion matrix table (as a NumPy array).
    *   `sns.heatmap()`: A function from the Seaborn library (built on Matplotlib) to draw the confusion matrix as a colored grid. `annot=True` shows the numbers in the cells, `fmt="d"` formats them as integers, `cmap='Blues'` sets the color scheme, and `xticklabels`/`yticklabels` add the actual class names to the axes based on our `labels` list. This makes the matrix much easier to read.

### Example Output (Based on Project Notebook Snippets)

The `classification_report` output from the project notebook looks like this (note the notebook might have 3 classes, but the concept is the same):

```
              precision    recall  f1-score   support

           0       0.96      0.79      0.87        96   -> Class 0 (e.g., Glioma)
           1       0.82      0.90      0.86        91   -> Class 1 (e.g., Meningioma)
           2       0.92      1.00      0.96        90   -> Class 2 (e.g., Pituitary)

    accuracy                           0.90       277   -> Overall Accuracy
   macro avg       0.90      0.90      0.89       277   -> Average across classes
weighted avg       0.90      0.90      0.89       277   -> Weighted average by class size
```

And the confusion matrix heatmap (simplified version like in the notebook):

```mermaid
pie title Confusion Matrix (Simplified)
    "Correct (Class 0)": 76
    "Mispredicted (Class 0 -> 1)": 10
    "Mispredicted (Class 0 -> 2)": 10
    "Mispredicted (Class 1 -> 0)": 7
    "Correct (Class 1)": 82
    "Mispredicted (Class 1 -> 2)": 2
    "Mispredicted (Class 2 -> 0)": 0
    "Mispredicted (Class 2 -> 1)": 0
    "Correct (Class 2)": 90

    Note: This is not a standard confusion matrix visualization, but represents the idea.
    A heatmap table (as shown in the Python code) is the standard.
```
*   **Reading the Heatmap:**
    *   Look at the diagonal cells (where True Label == Predicted Label). These numbers show how many images of that type were correctly classified. For example, 76 of the actual 'Glioma' tumors were predicted as 'Glioma'.
    *   Look at the off-diagonal cells. These show misclassifications. For example, 10 actual 'Glioma' tumors were misclassified as 'Meningioma' (True Label 'Glioma', Predicted Label 'Meningioma'). 7 actual 'Meningioma' tumors were misclassified as 'Glioma'.
    *   This detailed view tells you which classes are most often confused with each other. In this example, Class 2 ('Pituitary') seems to be classified perfectly (100% recall and precision for that class based on the confusion matrix numbers).

## How Evaluation Metrics are Calculated (Under the Hood)

Let's visualize the high-level process of getting these metrics after training:

```mermaid
sequenceDiagram
    participant Test Data (X_test, y_test)
    participant CNN Model
    participant Keras (model.evaluate)
    participant Scikit-learn (metrics)
    participant Evaluation Results

    Test Data (X_test, y_test)-->>CNN Model: Input test images (X_test)
    CNN Model->>CNN Model: Process images
    CNN Model-->>Scikit-learn (metrics): Output Predictions (Probabilities)
    Scikit-learn (metrics)->>Scikit-learn (metrics): Convert probabilities to class indices (e.g., using argmax)
    Test Data (X_test, y_test)-->>Scikit-learn (metrics): Provide True Labels (y_test, converted)

    Scikit-learn (metrics)->>Scikit-learn (metrics): Calculate Confusion Matrix (compare predicted vs true indices)
    Scikit-learn (metrics)->>Scikit-learn (metrics): Calculate Precision, Recall, F1-score (from Confusion Matrix)
    Scikit-learn (metrics)-->>Evaluation Results: Output Report & Matrix

    Keras (model.evaluate)->>CNN Model: Input test images (X_test)
    CNN Model->>CNN Model: Process images
    CNN Model-->>Keras (model.evaluate): Output Predictions (Probabilities)
    Keras (model.evaluate)->>Test Data (X_test, y_test): Provide True Labels (y_test)
    Keras (model.evaluate)->>Keras (model.evaluate): Calculate Loss and Overall Accuracy
    Keras (model.evaluate)-->>Evaluation Results: Output Loss & Accuracy

    Note over Evaluation Results: Combined, these give a full picture of performance.
```
*   **Explanation:** The test data is the input. The model makes predictions. These predictions are then compared against the *actual* correct labels for the test data. Keras handles the overall loss and accuracy calculation during `model.evaluate()`. For the more detailed metrics, we typically extract the predictions and true labels and use a separate library like Scikit-learn to compute the classification report and confusion matrix. All these results together provide a comprehensive understanding of the model's performance on unseen data.

## Why These Metrics Matter

Using a combination of these metrics is essential:

*   Overall accuracy is good for a quick check, but the classification report and confusion matrix are crucial for understanding the model's strengths and weaknesses across different classes.
*   For medical image classification like brain tumors, false negatives (missing a tumor) or false positives (incorrectly identifying a tumor in a healthy scan) can have serious consequences. Precision and Recall help quantify these specific types of errors for each tumor type.
*   The confusion matrix visually highlights which classes are most problematic and often gives clues for improvement (e.g., if two tumor types are constantly confused, perhaps they share very similar visual features).

By evaluating our model rigorously on a separate test set using multiple metrics, we gain confidence in its ability to generalize to new patient scans and understand its limitations.

## Conclusion

In this chapter, we learned the critical importance of evaluating our trained CNN model on a separate testing dataset to get an unbiased measure of its performance. We explored key evaluation metrics like overall accuracy, precision, recall, F1-score, and the confusion matrix, understanding what each metric tells us about the model's ability to classify different brain tumor types. We saw how to use Keras and scikit-learn to compute these metrics and visualize the confusion matrix.

Now that we have a trained and evaluated model, we're ready to put it to work and use it to make predictions on new, individual brain scan images.

[Image Prediction](08_image_prediction_.md)
