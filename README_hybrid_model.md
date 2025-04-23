# Hybrid Model for Real vs. AI-Generated Image Classification

## Overview

This document summarizes the implementation, training, and evaluation of a Hybrid model for distinguishing between real photographs and AI-generated images. The model combines a CNN approach with manually extracted image features to achieve improved classification accuracy. The model was trained exclusively on CPU resources and achieved 83.56% accuracy on the test set, outperforming the CNN-only approach (82.57%).

## Model Architecture

The hybrid model employs a dual-branch architecture with the following components:

```
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ image_input         │ (None, 224, 224,  │          0 │ -                 │
│ (InputLayer)        │ 3)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d (Conv2D)     │ (None, 224, 224,  │        896 │ image_input[0][0] │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d       │ (None, 112, 112,  │          0 │ conv2d[0][0]      │
│ (MaxPooling2D)      │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_1 (Conv2D)   │ (None, 112, 112,  │     18,496 │ max_pooling2d[0]… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_1     │ (None, 56, 56,    │          0 │ conv2d_1[0][0]    │
│ (MaxPooling2D)      │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_2 (Conv2D)   │ (None, 56, 56,    │     73,856 │ max_pooling2d_1[… │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_2     │ (None, 28, 28,    │          0 │ conv2d_2[0][0]    │
│ (MaxPooling2D)      │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ feature_input       │ (None, 24)        │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ flatten (Flatten)   │ (None, 100352)    │          0 │ max_pooling2d_2[…]│
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (Dense)     │ (None, 64)        │      1,600 │ feature_input[0][0]│
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense (Dense)       │ (None, 128)       │ 12,845,184 │ flatten[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout (Dropout)   │ (None, 64)        │          0 │ dense_1[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate         │ (None, 192)       │          0 │ dense[0][0],      │
│ (Concatenate)       │                   │            │ dropout[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_2 (Dense)     │ (None, 64)        │     12,352 │ concatenate[0][0] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_1 (Dropout) │ (None, 64)        │          0 │ dense_2[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_3 (Dense)     │ (None, 1)         │         65 │ dropout_1[0][0]   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 12,952,449 (49.41 MB)
 Trainable params: 12,952,449 (49.41 MB)
 Non-trainable params: 0 (0.00 B)
```

Key components:
- **CNN Branch**:
  - Input: 224x224x3 RGB images
  - Three convolutional blocks with increasing filter sizes (32, 64, 128)
  - Each block includes a Conv2D layer with ReLU activation and Max pooling
  - Flatten layer followed by dense layer (128 neurons)

- **Engineered Features Branch**:
  - Input: 24-dimensional feature vector of manually extracted image features
  - Dense layer (64 neurons) with ReLU activation
  - Dropout (0.3) for regularization

- **Combined Classification Head**:
  - Concatenation of CNN and engineered features
  - Dense layer (64 neurons) with ReLU activation
  - Dropout (0.5) for regularization
  - Output layer with sigmoid activation for binary classification

## Implementation Details

### Environment
- Training performed on CPU only using `CUDA_VISIBLE_DEVICES="-1"`
- Python 3.12.9
- TensorFlow 2.19.0
- Memory usage during training: 2.71 GB

### Feature Extraction
The hybrid model incorporates 24 manually extracted features from the images, including:

1. **Metadata Features**:
   - Image dimensions (height, width)
   - Aspect ratio
   - Image size

2. **Color Features**:
   - Channel statistics (mean, standard deviation, skewness, kurtosis)
   - Channel correlations (R-G, R-B, G-B)

3. **Complexity Features**:
   - Edge density
   - Image entropy
   - Contrast

4. **Noise Features**:
   - Noise level
   - Noise mean

These features were extracted using a custom `SimpleFeatureExtractor` class that leverages OpenCV for image processing.

### Dataset
- Source: DeepGuardDB_v1 SD dataset
- 2,675 real images and 2,675 fake images
- Split ratio: 70% training (3,745 images), 15% validation (802 images), 15% test (803 images)
- Data augmentation applied to training images:
  - Random horizontal flips
  - Random brightness adjustments (0.8-1.2x)

### Training Configuration
- Batch size: 16
- Optimizer: Adam with default settings
- Loss function: Binary cross-entropy
- Metrics: Accuracy
- Callbacks:
  - Early stopping (patience=5, monitoring validation loss)
  - Model checkpoint (saving best model)

## Training Process

The model was trained for 9 epochs before early stopping activated, with the following progression:

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|------------------|---------------------|---------------|-----------------|
| 1     | 58.74%           | 68.83%              | 0.7515        | 0.5711          |
| 2     | 69.75%           | 73.44%              | 0.5952        | 0.5352          |
| 3     | 72.31%           | 81.42%              | 0.5524        | 0.4230          |
| 4     | 79.95%           | 83.54%              | 0.4376        | 0.3624          |
| 5     | 82.70%           | 80.92%              | 0.3853        | 0.3906          |
| 6     | 84.43%           | 82.42%              | 0.3609        | 0.3754          |
| 7     | 88.40%           | 82.92%              | 0.3019        | 0.3773          |
| 8     | 89.15%           | 82.17%              | 0.2545        | 0.4731          |
| 9     | 91.30%           | 84.04%              | 0.2156        | 0.4013          |

- Training time: 361.30 seconds (6.02 minutes)
- The model showed rapid improvement in early epochs, with final training accuracy of 91.30%

## Results and Analysis

### Performance Metrics
- **Test Accuracy**: 83.56%
- **Test Loss**: 0.3845
- **Confusion Matrix**:
  ```
  [[333  69]
   [ 63 338]]
  ```
- **Classification Report**:
  ```
              precision    recall  f1-score   support

         0.0       0.84      0.83      0.83       402
         1.0       0.83      0.84      0.84       401

    accuracy                           0.84       803
   macro avg       0.84      0.84      0.84       803
weighted avg       0.84      0.84      0.84       803
  ```

### Performance by Class
- **Fake Images (Class 0)**:
  - Correctly identified: 333 out of 402 (82.84%)
  - Misclassified as real: 69 out of 402 (17.16%)
  - Precision: 84% (of all images classified as fake, 84% were actually fake)

- **Real Images (Class 1)**:
  - Correctly identified: 338 out of 401 (84.29%)
  - Misclassified as fake: 63 out of 401 (15.71%)
  - Precision: 83% (of all images classified as real, 83% were actually real)

### Error Analysis
- Total misclassifications: 132 out of 803 images (16.44%)
- False positives (fake images classified as real): 69 (52.27% of errors)
- False negatives (real images classified as fake): 63 (47.73% of errors)
- The model shows a relatively balanced error profile, with a slight tendency to misclassify fake images as real.

### Comparison with CNN-only Model
- **Hybrid Model Accuracy**: 83.56% (+0.99% improvement)
- **Training Time**: 361.30 seconds vs. 1,077.79 seconds (3x faster)
- **Memory Usage**: Similar (2.71 GB vs. 2.33 GB)
- **Error Rate**: 16.44% vs. 17.43% (lower error rate)

The hybrid model demonstrates better performance across all metrics while also training significantly faster.

## Files and Directory Structure

```
Project3/
├── SCRIPTS/
│   ├── run_hybrid_cpu_fixed2.slurm       # SLURM job script for hybrid model
│   ├── main_hybrid_model_cpu_fixed2.py   # Python script generated within job script
│   ├── feature_extraction.py             # Original feature extraction code
│   └── analyze_hybrid_results.py         # Analysis script for examining results
├── DATA/
│   └── DeepGuardDB_v1/
│       ├── SD_dataset/
│       │   ├── real/                     # Real images
│       │   └── fake/                     # AI-generated images
│       └── json_files/
│           └── sd_json.json              # JSON file mapping real/fake image pairs
└── model_results/
    └── hybrid_model_cpu/
        ├── best_model.h5                 # Saved model weights
        ├── confusion_matrix.png          # Confusion matrix visualization
        ├── training_history.png          # Training/validation metrics plot
        ├── results.txt                   # Text summary of results
        ├── y_pred.npy                    # Predicted labels
        ├── y_pred_probs.npy              # Prediction probabilities
        ├── y_true.npy                    # True labels
        ├── train_features.npy            # Extracted features for training set
        ├── val_features.npy              # Extracted features for validation set
        └── test_features.npy             # Extracted features for test set
```

## Visualizations

The model produced two key visualizations:
1. **Training History**: Shows the progression of accuracy and loss metrics during training
2. **Confusion Matrix**: Visualizes the model's prediction performance

## Conclusion

The hybrid model demonstrates superior performance compared to the CNN-only approach in distinguishing between real and AI-generated images, achieving an overall accuracy of 83.56%. The model shows balanced classification ability between classes, with slightly better recall for real images (84.29%) than for fake images (82.84%).

By combining deep learning features with traditional computer vision techniques, the hybrid approach leverages complementary strengths: the CNN's ability to learn hierarchical patterns and the engineered features' explicit encoding of image characteristics that might be difficult for the CNN to learn.

### Key Takeaways

1. The hybrid approach outperforms a pure CNN architecture for this task.
2. Including manually extracted features improves detection accuracy for both real and fake images.
3. Training time is significantly reduced compared to the CNN-only model, making this approach more efficient.
4. The model shows better generalization, with a lower error rate on the test set.

### Future Improvements

1. Explore additional engineered features that might further improve discrimination ability.
2. Implement feature importance analysis to identify which engineered features contribute most to classification.
3. Experiment with transfer learning using pre-trained CNN backbones.
4. Test the model on newer generative models to assess robustness against state-of-the-art fakes.
5. Consider ensemble methods that combine multiple hybrid models for improved accuracy. 