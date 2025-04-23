# CNN Model for Real vs. AI-Generated Image Classification

## Overview

This document summarizes the implementation, training, and evaluation of a Convolutional Neural Network (CNN) model for distinguishing between real photographs and AI-generated images. The model was trained exclusively on CPU resources and achieved 82.57% accuracy on the test set.

## Model Architecture

The model employs a standard CNN architecture with the following components:

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 224, 224, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 112, 112, 32)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 112, 112, 64)   │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 56, 56, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 56, 56, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 28, 28, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 100352)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │    12,845,184 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │           129 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 12,938,561 (49.36 MB)
 Trainable params: 12,938,561 (49.36 MB)
 Non-trainable params: 0 (0.00 B)
```

Key components:
- **Input**: 224x224x3 RGB images
- **Convolutional Blocks**: Three convolutional blocks, each consisting of:
  - Convolutional layer with increasing filter sizes (32, 64, 128)
  - ReLU activation
  - Max pooling layer (2x2)
- **Classification Head**:
  - Flatten layer
  - Fully connected layer (128 neurons)
  - Dropout (0.5) for regularization
  - Output layer with sigmoid activation for binary classification

## Implementation Details

### Environment
- Training performed on CPU only using `CUDA_VISIBLE_DEVICES="-1"`
- Python 3.12.9
- TensorFlow 2.19.0
- Memory usage: 2.33 GB

### Dataset
- Source: DeepGuardDB_v1 SD dataset
- 2,675 real images and 2,675 fake images
- Split ratio: 70% training (3,745 images), 15% validation (802 images), 15% test (803 images)
- Data augmentation applied to training set:
  - Random rotation (±10°)
  - Width/height shifts (±10%)
  - Horizontal flips

### Training Configuration
- Batch size: 16
- Learning rate: Default Adam optimizer settings
- Loss function: Binary cross-entropy
- Metrics: Accuracy
- Callbacks:
  - Early stopping (patience=5, monitoring validation loss)
  - Model checkpoint (saving best model)

## Training Process

The model was trained for 20 epochs, with the following progression:

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|------------------|---------------------|---------------|-----------------|
| 1     | 51.29%           | 59.60%              | 0.7698        | 0.6614          |
| 5     | 68.70%           | 74.94%              | 0.5808        | 0.5170          |
| 10    | 77.33%           | 78.80%              | 0.4750        | 0.4687          |
| 15    | 80.40%           | 81.42%              | 0.4272        | 0.4348          |
| 20    | 81.98%           | 82.29%              | 0.3993        | 0.4316          |

- Training time: 1,077.79 seconds (17.96 minutes)
- Training showed steady improvement without signs of overfitting

## Results and Analysis

### Performance Metrics
- **Test Accuracy**: 82.57%
- **Confusion Matrix**:
  ```
  [[324  78]
   [ 62 339]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support
  
           0.0       0.84      0.81      0.82       402
           1.0       0.81      0.85      0.83       401
  
      accuracy                           0.83       803
     macro avg       0.83      0.83      0.83       803
  weighted avg       0.83      0.83      0.83       803
  ```

### Performance by Class
- **Fake Images (Class 0)**:
  - Correctly identified: 324 out of 402 (80.60%)
  - Misclassified as real: 78 out of 402 (19.40%)
  - Precision: 84% (of all images classified as fake, 84% were actually fake)

- **Real Images (Class 1)**:
  - Correctly identified: 339 out of 401 (84.54%)
  - Misclassified as fake: 62 out of 401 (15.46%)
  - Precision: 81% (of all images classified as real, 81% were actually real)

### Error Analysis
- Total misclassifications: 140 out of 803 images (17.43%)
- False positives (fake images classified as real): 78 (55.71% of errors)
- False negatives (real images classified as fake): 62 (44.29% of errors)
- The model shows a slightly higher tendency to misclassify fake images as real than vice versa.

## Files and Directory Structure

```
Project3/
├── SCRIPTS/
│   ├── run_hybrid_cpu_real_fixed2.slurm  # SLURM job script for CPU-only model
│   └── main_hybrid_cpu_real_fixed2.py    # Python script generated within job script
├── DATA/
│   └── DeepGuardDB_v1/
│       ├── SD_dataset/
│       │   ├── real/                     # Real images
│       │   └── fake/                     # AI-generated images
│       └── json_files/
│           └── sd_json.json              # JSON file mapping real/fake image pairs
└── model_results/
    └── hybrid_cpu_real/
        ├── best_model.h5                 # Saved model weights
        ├── confusion_matrix.png          # Confusion matrix visualization
        ├── training_history.png          # Training/validation metrics plot
        ├── results.txt                   # Text summary of results
        ├── y_pred.npy                    # Predicted labels
        └── y_true.npy                    # True labels
```

## Visualizations

The model produced two key visualizations:
1. **Training History**: Shows the progression of accuracy and loss metrics during training
2. **Confusion Matrix**: Visualizes the model's prediction performance

## Conclusion

The CNN model demonstrates strong performance in distinguishing between real and AI-generated images, achieving an overall accuracy of 82.57%. The model shows balanced classification ability between classes, with slightly better recall for real images (85%) than for fake images (81%).

This experiment proves that effective AI image detection can be performed using CPU resources alone, which is valuable for applications where GPU resources may be limited or unavailable. The model's accuracy is notable given the challenging nature of the task and the exclusive use of CPU for training.

### Key Takeaways

1. A relatively simple CNN architecture can effectively identify AI-generated images with good accuracy.
2. The model performs slightly better at identifying real images compared to fake ones.
3. Training on CPU is feasible for this task, with reasonable training times (under 20 minutes).
4. The model showed consistent improvement during training without significant overfitting.

### Future Improvements

1. Experiment with deeper architectures or pre-trained models for feature extraction.
2. Incorporate more diverse augmentation techniques to improve generalization.
3. Explore ensemble methods to combine multiple models for improved accuracy.
4. Test on newer generative models to assess robustness against state-of-the-art fakes. 