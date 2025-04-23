# Project 3: Real vs. AI-Generated Image Classification

Our team name is Jarheads. Our theme song is "Eye of the Tiger" by Survivor. Our mascot is an english bulldog named Chesty.

## Project Overview

This repository contains implementation, training results, and performance analysis of three distinct deep learning models designed to distinguish between real photographs and AI-generated images:

1. **Hybrid Model**: A CPU-friendly model that uses pre-extracted features from images
2. **CNN Model**: A traditional Convolutional Neural Network that processes raw image data
3. **Transfer Model**: A simplified transfer learning approach that doesn't require downloading pre-trained weights

## Model Performance Summary

| Model | Test Accuracy | Test Loss | Training Time |
|-------|--------------|-----------|---------------|
| Hybrid Model | 83.56% | 0.3845 | 361.30 seconds |
| CNN Model | 82.57% | 0.4316 | 1,077.79 seconds |
| Transfer Model | 49.07% | 0.6942 | 3.15 seconds |

Key findings:
- The Hybrid Model achieves the best balance of accuracy and training time
- The CNN Model provides comparable accuracy but requires significantly more training time
- The Transfer Model offers extremely fast training but with much lower accuracy (near random chance)

## Visualizations

The repository includes several visualizations for model comparison:

- Accuracy comparison across models
- Loss comparison
- Training time analysis
- Model complexity comparison
- Radar chart showing multi-dimensional performance metrics
- Training history for the Transfer Model
- Visual representation of model architectures

All visualizations are located in the `visualization_results` directory and were generated using matplotlib and seaborn.

## Directory Structure

```
Project3/
├── SCRIPTS/                           # Training and evaluation scripts
│   ├── cnn_model.py                   # CNN model implementation
│   ├── hybrid_model.py                # Hybrid model implementation
│   ├── transfer_simple.py             # Transfer learning model
│   └── *.slurm                        # SLURM job scripts
├── model_results/                     # Results from model training
│   ├── cnn_updated/                   # CNN model results
│   ├── hybrid_cpu/                    # Hybrid model results
│   └── transfer_simple/               # Transfer model results
├── visualization_results/             # Generated visualizations
│   ├── accuracy_comparison.png
│   ├── architecture_comparison.png
│   ├── loss_comparison.png
│   ├── model_complexity.png
│   ├── radar_comparison.png
│   ├── time_vs_accuracy.png
│   ├── training_time_comparison.png
│   └── transfer_model_history.png
├── README.md                          # This file
├── README_CNN.md                      # Detailed CNN model documentation
├── README_Transfer_Model.md           # Detailed Transfer model documentation
└── README_hybrid_model.md             # Detailed Hybrid model documentation
```

## Running the Visualization Generator

To generate the visualization charts:

```bash
python generate_model_visualizations.py
```

This will create/update all charts in the `visualization_results` directory.

## Model Documentation

For detailed information about each model, refer to the following files:
- [Hybrid Model Documentation](./README_hybrid_model.md)
- [CNN Model Documentation](./README_CNN.md)
- [Transfer Model Documentation](./README_Transfer_Model.md)
