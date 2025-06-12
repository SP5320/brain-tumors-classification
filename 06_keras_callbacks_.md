# Chapter 6: Keras Callbacks

Welcome back! In [Chapter 5: Model Training](05_model_training_.md), we dove into the process of teaching our CNN model to classify brain tumor images. We compiled the model, defined the optimizer and loss function, and used `model.fit()` to start the learning loop over several epochs.

Training a deep learning model can take a while, often for many epochs. During this process, you might want the training to adapt based on how well the model is performing. For example:
*   What if the model stops improving on the validation data? Continuing to train might lead to overfitting.
*   What if the model's progress slows down significantly? Maybe adjusting the learning rate could help it find a better solution.
*   What if you want to automatically save the model every time it achieves the best performance so far?

Stopping and restarting training manually to check these things would be incredibly tedious and inefficient. This is where **Keras Callbacks** come to the rescue!

## What are Keras Callbacks?

Imagine you have a student studying for a big exam (this is your CNN training). You, the teacher, want to monitor their progress closely. Instead of constantly looking over their shoulder, you hire a **personal assistant** (the callback) who can:

*   Check their practice test scores after each study session (at the end of an epoch).
*   Notify you if they haven't improved for a few sessions.
*   Suggest a change in study strategy if they get stuck (like reducing the learning rate).
*   Make a copy of their notes whenever they achieve a new personal best score.

Keras Callbacks are essentially these "personal assistants" for your training process. They are special objects or functions that you can hook into your model's training loop to perform actions at specific points:
*   At the start or end of training.
*   At the start or end of each epoch.
*   At the start or end of each training batch.

By using callbacks, you automate monitoring and decision-making during training, making the process more efficient and potentially leading to a better model.

## Why Use Callbacks?

Using callbacks provides significant advantages:

1.  **Automation:** They automate tasks that you would otherwise have to do manually (like checking validation accuracy or saving models).
2.  **Improved Convergence:** Callbacks like `ReduceLROnPlateau` can dynamically adjust training parameters (like the learning rate) to help the model converge better and faster.
3.  **Preventing Overfitting:** Callbacks like `EarlyStopping` can monitor validation performance and stop training automatically if the model starts overfitting (i.e., getting better on training data but worse on unseen validation data).
4.  **Better Monitoring & Debugging:** Callbacks like `TensorBoard` provide detailed logs and visualizations of the training process, which are invaluable for understanding how your model is learning and debugging issues.
5.  **Saving Progress:** `ModelCheckpoint` allows you to automatically save your model's weights periodically or when a specific performance metric improves.

## Focus on `ReduceLROnPlateau`

In our brain tumor classification project code ([`classification.py`](classification.py) or [`compiled_layered_model.ipynb`](compiled_layered_model.ipynb)), one important callback used is `ReduceLROnPlateau`. Let's understand this one in detail, as it's a very common and useful callback.

**The Problem:** During training, the learning rate determines how big of a step the optimizer takes to adjust the model's weights ([Chapter 5: Model Training](05_model_training_.md)). A high learning rate can make training faster but might overshoot the optimal solution. A low learning rate can be more stable but might get stuck or take too long. Finding the perfect learning rate is tricky, and it often needs to be adjusted *during* training.

**The Solution: `ReduceLROnPlateau`**

This callback monitors a chosen metric (like validation accuracy or validation loss). If the metric doesn't improve for a certain number of epochs (`patience`), the callback automatically reduces the learning rate by a specified `factor`. The idea is that if the model's learning has plateaued (stopped improving), reducing the learning rate might help it navigate more carefully to find a better minimum in the loss landscape.

Let's look at the key parameters:

| Parameter       | Description                                                                 | Analogy (Student Studying)                                 |
| :-------------- | :-------------------------------------------------------------------------- | :--------------------------------------------------------- |
| `monitor`       | The metric to track (e.g., 'val_accuracy' or 'val_loss').                   | Which score to watch (e.g., practice test grade).          |
| `factor`        | The factor by which the learning rate will be reduced (e.g., 0.1 means LR becomes LR * 0.1). | How much to reduce the study intensity (e.g., study 10% less hours). |
| `patience`      | Number of epochs with no improvement on the monitored metric after which learning rate will be reduced. | How many practice tests must be taken with no score improvement. |
| `min_delta`     | Minimum change in the monitored metric to qualify as an improvement.        | How small of a score increase is considered "improvement". |
| `mode`          | 'auto', 'min', or 'max'. If 'min', LR is reduced when monitored metric stops decreasing. If 'max', when it stops increasing. 'auto' guesses based on the metric name. | Whether a *higher* score is better ('max' for accuracy) or *lower* ('min' for loss). |
| `verbose`       | Integer. 0: silent, 1: messages when LR is reduced.                         | Whether the assistant announces LR changes out loud.       |

## How Callbacks are Used in Keras Code

Callbacks are passed to the `model.fit()` method using the `callbacks` argument, which expects a list of callback objects.

From [Chapter 5: Model Training](05_model_training_.md), we saw the `model.fit()` call that looked like this:

```python
# ... model compilation and data preparation ...

tensorboard = TensorBoard(log_dir = 'logs')
# checkpoint = ModelCheckpoint("vgg16.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1) # Another common callback (commented out in the code)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)

history=model.fit(
    X_train,y_train,
    validation_split=0.1, # Note: Redundant when validation_data is provided, which it is below.
    epochs =20,
    verbose=1,
    batch_size=32,
    validation_data=(X_test, y_test), # This is the validation data Keras uses
    callbacks=[tensorboard, reduce_lr] # Here's where the callbacks are passed!
)
```

*   **Explanation:**
    *   We first create instances of the callbacks we want to use: `tensorboard` (for logging) and `reduce_lr` (our `ReduceLROnPlateau`).
    *   We configure `reduce_lr` to `monitor='val_accuracy'`. This means it will watch the accuracy on the validation data (`validation_data=(X_test, y_test)`).
    *   `factor=0.3` means the learning rate will be multiplied by 0.3 (reduced to 30% of its current value).
    *   `patience=2` means it will wait for 2 epochs without improvement in validation accuracy before reducing the learning rate.
    *   `min_delta=0.001` means an improvement must be at least 0.001 (0.1%) to count as improvement.
    *   `mode='auto'` is appropriate because 'val_accuracy' should increase (maximize) for better performance.
    *   `verbose=1` will print a message to the console every time the learning rate is reduced.
    *   Finally, the `callbacks=[tensorboard, reduce_lr]` list is passed to `model.fit()`. Keras knows to activate these callbacks at the appropriate times during training.

Let's look specifically at the `ReduceLROnPlateau` definition from the code:

```python
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)
```

*   **Explanation:** As detailed above, this line creates the callback object configured to monitor validation accuracy, reduce the learning rate by a factor of 0.3 if validation accuracy doesn't improve by at least 0.001 for 2 consecutive epochs, and print messages when this happens.

## How Callbacks Work (Under the Hood)

Here's a simplified view of how a callback like `ReduceLROnPlateau` fits into the training process, focusing on the end of an epoch:

```mermaid
sequenceDiagram
    participant Training Loop (model.fit)
    participant CNN Model
    participant Keras Callbacks (e.g., reduce_lr)
    participant Optimizer
    participant Training Metrics

    Training Loop (model.fit)->>CNN Model: Train on a batch...
    CNN Model-->>Training Loop (model.fit): Update weights

    Note over Training Loop (model.fit): ...Repeat for all batches (one epoch)...

    Training Loop (model.fit)->>CNN Model: Evaluate on validation data
    CNN Model-->>Training Metrics: Return validation loss/accuracy
    Training Metrics-->>Training Loop (model.fit): Pass metrics for the epoch

    Training Loop (model.fit)->>Keras Callbacks (e.g., reduce_lr): Call callbacks (e.g., `on_epoch_end`)
    Keras Callbacks (e.g., reduce_lr)->>Training Metrics: Check monitored metric (e.g., val_accuracy)
    Keras Callbacks (e.g., reduce_lr)->>Keras Callbacks (e.g., reduce_lr): Check if metric improved (vs patience)
    alt If no improvement after patience
        Keras Callbacks (e.g., reduce_lr)->>Optimizer: Request learning rate reduction
        Optimizer->>Optimizer: Adjust learning rate
        Keras Callbacks (e.g., reduce_lr)->>Training Loop (model.fit): Signal LR change (if verbose)
    end
    Keras Callbacks (e.g., reduce_lr)-->>Training Loop (model.fit): Return control

    Note over Training Loop (model.fit): Start next epoch (possibly with new LR)
```

*   **Explanation:** After each epoch finishes and the model has been evaluated on the validation data, Keras triggers the callbacks you provided. The `ReduceLROnPlateau` callback looks at the validation accuracy from that epoch. It compares it to the best validation accuracy seen so far. If there hasn't been a significant improvement (based on `min_delta`) for `patience` number of epochs, the callback calculates the new learning rate (current LR * `factor`) and tells the optimizer to update its learning rate accordingly for the subsequent epochs.

The training log shown in [Chapter 5](05_model_training_.md) includes output messages like `Epoch 7: ReduceLROnPlateau reducing learning rate...`, confirming that this callback was active and made adjustments during training.

## Other Useful Callbacks

While `ReduceLROnPlateau` is highlighted because it's in the project code, other callbacks are very common in deep learning:

| Callback Name       | Purpose                                                             | Benefit                                         |
| :------------------ | :------------------------------------------------------------------ | :---------------------------------------------- |
| `EarlyStopping`     | Stop training when a monitored metric has stopped improving.        | Prevents overfitting, saves training time.      |
| `ModelCheckpoint`   | Save the model or its weights after each epoch or when a metric improves. | Easily recover the best model, backup progress. |
| `TensorBoard`       | Log metrics and model graphs for visualization in the TensorBoard UI. | Powerful monitoring, debugging, and analysis.   |

You can use multiple callbacks simultaneously, as shown in the `model.fit()` call in the project code, which uses both `TensorBoard` and `ReduceLROnPlateau`.

## Conclusion

In this chapter, we learned about Keras Callbacks, powerful tools that act as assistants during the model training process. We saw how they can automate tasks like monitoring performance and adjusting the learning rate, helping to improve model convergence and prevent overfitting. We focused on the `ReduceLROnPlateau` callback used in our project code and understood how it works to adapt the learning rate dynamically.

Now that our model is trained, we need to properly assess its performance. The next chapter will cover how to evaluate our model and understand key metrics for classification.

[Model Evaluation and Metrics](07_model_evaluation_and_metrics_.md)
