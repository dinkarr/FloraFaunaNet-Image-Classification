# FloraFaunaNet-Image-Classification

FloraFaunaNet is an image classification model fine-tuned to classify images into 10 classes of flora and fauna. This model leverages the Vision Transformer (ViT) architecture and utilizes various data augmentation techniques to enhance the generalization capability of the model.

## Objectives

- Fine-tune the pre-trained Vision Transformer (ViT) model for the flora and fauna image classification task.
- Implement various data augmentation techniques to enhance model robustness and prevent overfitting.
- Evaluate the model performance using various metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.

## Data Augmentation Techniques

To enhance the training process and help the model generalize better, the following data augmentation techniques were applied to the training dataset:

1. **Resize**: Images were resized to a consistent size of 224x224 pixels using bicubic interpolation.
  
2. **Random Horizontal Flip**: Applied with a 50% probability to flip images horizontally, simulating variations in object orientation.

3. **Random Vertical Flip**: Applied with a 50% probability to flip images vertically, further improving robustness.

4. **Random Rotation**: Images were rotated by up to 30 degrees randomly to simulate different orientations in real-world scenarios.

5. **Random Crop with Padding**: A random crop of size 224x224 was applied, with 4 pixels of padding, to aid the model in becoming invariant to the position of objects within the image.

6. **Color Jitter**: 
   - **Brightness**: Adjusted by a factor of 0.5.
   - **Contrast**: Adjusted by a factor of 0.5.
   - **Saturation**: Adjusted by a factor of 0.5.
   - **Hue**: Adjusted by a factor of 0.1.

   This introduced variation in color properties, helping the model generalize under varying lighting conditions.

7. **Normalization**: The images were normalized using the ImageNet mean and standard deviation values to ensure that pixel values fall within a consistent range.

## Fine-Tuning Process

- **Pre-trained Model**: The Vision Transformer (ViT) model (`vit_h_14` variant) was used as the base model, pre-trained on the ImageNet dataset.
- **Model Modification**: The final layers of the model were replaced:
  - A new fully connected layer with 128 output features and a batch normalization layer.
  - A GELU activation function and Dropout (0.25) were added for regularization.
  - The output layer was updated to have 10 classes for the flora and fauna classification task.
  
- **Freezing Pre-trained Layers**: Initially, the pre-trained layers were frozen to prevent updates during training, focusing only on the new classification head.
- **Fine-Tuning Specific Layers**: The final classification head and some specific layers in the encoder were fine-tuned to optimize for the flora and fauna classification task.

## Evaluation Metrics

The model was evaluated using the following metrics to assess its classification performance:

- **Accuracy**: The overall percentage of correct predictions.
- **F1-Score**: The harmonic mean of precision and recall, providing a balanced evaluation.
- **Precision**: The proportion of positive predictions that were actually correct.
- **Recall**: The proportion of actual positives that were correctly identified.
- **AUC-ROC**: The area under the Receiver Operating Characteristic curve, evaluated for multiclass classification.

## Evaluation Results

During evaluation, the model was tested on the validation set to measure its performance across various metrics. The following results were obtained:

- **Accuracy**: A high level of accuracy on the validation dataset, indicating strong classification performance.
- **F1-Score**: Demonstrated a balanced precision and recall across different classes.
- **AUC-ROC**: A good ROC curve score, showing the model’s capability to discriminate between different classes.

## Running the Model

1. **Set up the environment**: Ensure that you have installed all required libraries and dependencies.
2. **Load the dataset**: The dataset should be organized into training and validation folders.
3. **Run the training script**: Fine-tuning is performed by updating only the last few layers, leveraging pre-trained weights from the ViT model.
4. **Evaluate the model**: After training, evaluate the model on the validation set to check its performance using various metrics.

## Conclusion

The FloraFaunaNet model, using Vision Transformer and fine-tuned with data augmentation techniques, performs well on the flora and fauna classification task. By employing advanced data augmentation strategies, the model's robustness is significantly improved, ensuring better generalization on unseen data.
