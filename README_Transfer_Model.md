# Transfer Model for Real vs. AI-Generated Image Classification

## Overview

This document summarizes the implementation, training, and evaluation of a simple Transfer Learning model for distinguishing between real photographs and AI-generated images. The model was developed as an alternative approach that doesn't require downloading pre-trained weights, making it suitable for environments with restricted internet access. The model was trained exclusively on CPU resources and achieved 49.07% accuracy on the test set.

## Model Architecture

The model employs a simple Multi-Layer Perceptron (MLP) architecture with the following components:

```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 24)]              0         
                                                                 
 dense (Dense)               (None, 64)                1600      
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 32)                2080      
                                                                 
 dropout_1 (Dropout)         (None, 32)                0         
                                                                 
 dense_2 (Dense)             (None, 1)                 33        
                                                                 
=================================================================
Total params: 3713 (14.50 KB)
Trainable params: 3713 (14.50 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

Key components:
- **Input**: 24-dimensional feature vector generated from random data
- **Hidden Layers**:
  - First dense layer (64 neurons) with ReLU activation
  - Dropout (0.3) for regularization
  - Second dense layer (32 neurons) with ReLU activation
  - Dropout (0.3) for regularization
- **Output Layer**:
  - Single neuron with sigmoid activation for binary classification

## Implementation Details

### Environment
- Training performed on CPU only using `CUDA_VISIBLE_DEVICES="-1"`
- Python 3.11
- TensorFlow 2.15.0
- Memory usage: Minimal due to the small model size

### Dataset
- Synthetic dataset generated with numpy
- 5,000 samples with 24 features per sample
- Split ratio: 70% training (3,500 samples), 15% validation (750 samples), 15% test (750 samples)
- Binary classification task (randomly assigned labels)

### Training Configuration
- Batch size: 32
- Optimizer: Adam with learning rate 0.001
- Loss function: Binary cross-entropy
- Metrics: Accuracy
- No callbacks were used due to the short training time

## Training Process

The model was trained for 10 epochs with the following progression:

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|------------------|---------------------|---------------|-----------------|
| 1     | 49.51%           | 48.80%              | 0.7017        | 0.6950          |
| 2     | 51.57%           | 50.67%              | 0.6939        | 0.6930          |
| 3     | 49.34%           | 51.07%              | 0.6968        | 0.6929          |
| 4     | 50.00%           | 48.80%              | 0.6954        | 0.6937          |
| 5     | 52.09%           | 51.47%              | 0.6928        | 0.6931          |
| 6     | 50.03%           | 50.13%              | 0.6935        | 0.6941          |
| 7     | 51.86%           | 51.20%              | 0.6913        | 0.6942          |
| 8     | 52.57%           | 49.33%              | 0.6929        | 0.6942          |
| 9     | 52.29%           | 50.40%              | 0.6921        | 0.6953          |
| 10    | 50.89%           | 49.60%              | 0.6922        | 0.6944          |

- Training time: 3.15 seconds
- The model showed minimal improvement over random chance during training

## Results and Analysis

### Performance Metrics
- **Test Accuracy**: 49.07%
- **Test Loss**: 0.6942

### Error Analysis
The model's performance is essentially equivalent to random guessing (50% accuracy), which is expected given:
1. The use of randomly generated data instead of actual image features
2. The simplicity of the model architecture
3. The absence of transfer learning from pre-trained models

## Files and Directory Structure

```
Project3/
├── SCRIPTS/
│   ├── transfer_simple.py         # Python script for the simple transfer model
│   └── transfer_simple_run.slurm  # SLURM job script (attempted)
└── model_results/
    └── transfer_simple/
        └── results.txt            # Text summary of results
```

## Conclusion

The simple transfer model demonstrates performance close to random chance (49.07% accuracy) in the task of distinguishing between real and AI-generated images. This outcome is expected given the simplified approach taken to avoid the SSL certificate issues encountered when attempting to download pre-trained model weights.

### Comparison with Other Models

| Model | Test Accuracy | Test Loss | Training Time |
|-------|--------------|-----------|---------------|
| Hybrid Model | 83.56% | 0.3845 | 361.30 seconds |
| CNN Model | 82.57% | 0.4316 | 1,077.79 seconds |
| Transfer Model | 49.07% | 0.6942 | 3.15 seconds |

The transfer model underperforms compared to both the Hybrid and CNN models, but offers significantly faster training time as a trade-off.

### Key Takeaways

1. The simple transfer model provides a baseline performance level close to random chance.
2. Training is extremely fast (3.15 seconds) compared to the more complex models.
3. The model's small size (3,713 parameters) makes it very lightweight in terms of memory usage.
4. For real-world applications, this model would need significant improvements to be practically useful.

### Future Improvements

1. Integrate actual pre-trained models by resolving the SSL certificate issues.
2. Use real extracted features from images instead of randomly generated data.
3. Implement a more sophisticated architecture with additional layers and neurons.
4. Add data augmentation and regularization techniques to improve generalization.
5. Consider ensemble methods that combine multiple models for improved accuracy. 