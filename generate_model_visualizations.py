import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Create output directory if it doesn't exist
os.makedirs('visualization_results', exist_ok=True)

# Model comparison data
models = ['Hybrid Model', 'CNN Model', 'Transfer Model']
accuracy = [83.56, 82.57, 49.07]  # in percentage
test_loss = [0.3845, 0.4316, 0.6942]
training_time = [361.30, 1077.79, 3.15]  # in seconds

# Set the style
plt.style.use('ggplot')
sns.set_palette("Set2")

# Figure 1: Test Accuracy Comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Test Accuracy Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.2f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_results/accuracy_comparison.png', dpi=300)
plt.close()

# Figure 2: Test Loss Comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(models, test_loss, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Test Loss Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Loss Value', fontsize=14)
plt.ylim(0, max(test_loss) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_results/loss_comparison.png', dpi=300)
plt.close()

# Figure 3: Training Time Comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(models, training_time, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Training Time Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Time (seconds)', fontsize=14)
plt.ylim(0, max(training_time) * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{height:.2f}s', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_results/training_time_comparison.png', dpi=300)
plt.close()

# Figure 4: Training Time vs Accuracy Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(training_time, accuracy, s=200, alpha=0.7, c=['#3498db', '#2ecc71', '#e74c3c'])

# Add model names as labels
for i, model in enumerate(models):
    plt.annotate(model, (training_time[i], accuracy[i]), 
                 xytext=(10, 5), textcoords='offset points',
                 fontsize=12, fontweight='bold')

plt.title('Training Time vs Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Training Time (seconds)', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualization_results/time_vs_accuracy.png', dpi=300)
plt.close()

# Figure 5: Model Complexity Chart (using parameters as proxy)
model_params = [3713, 61922, 3713]  # Hybrid, CNN, Transfer (using Transfer param count for Hybrid as placeholder)
plt.figure(figsize=(10, 6))
bars = plt.bar(models, model_params, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Model Complexity Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Number of Parameters', fontsize=14)
plt.ylim(0, max(model_params) * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1000,
            f'{height:,}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_results/model_complexity.png', dpi=300)
plt.close()

# Figure 6: Radar Chart comparing models
categories = ['Accuracy', 'Speed', 'Simplicity', 'Loss Performance', 'Parameter Efficiency']
N = len(categories)

# Normalize values between 0 and 1 for radar chart
# For accuracy: higher is better
accuracy_norm = [val/100 for val in accuracy]
# For training_time: lower is better, so we invert
speed_norm = [1 - (t / max(training_time)) for t in training_time]
# For simplicity: lower params is simpler
simplicity_norm = [1 - (p / max(model_params)) for p in model_params]
# For loss: lower is better, so we invert
loss_norm = [1 - (l / max(test_loss)) for l in test_loss]
# For parameter efficiency: accuracy per parameter (higher is better)
param_efficiency = [acc / (param/1000) for acc, param in zip(accuracy, model_params)]
param_efficiency_norm = [p / max(param_efficiency) for p in param_efficiency]

# Combine the data
values = [
    [accuracy_norm[0], speed_norm[0], simplicity_norm[0], loss_norm[0], param_efficiency_norm[0]],  # Hybrid
    [accuracy_norm[1], speed_norm[1], simplicity_norm[1], loss_norm[1], param_efficiency_norm[1]],  # CNN
    [accuracy_norm[2], speed_norm[2], simplicity_norm[2], loss_norm[2], param_efficiency_norm[2]]   # Transfer
]

# Angle for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Add categories
plt.xticks(angles[:-1], categories, fontsize=12)

# Draw each model
for i, model in enumerate(models):
    values_model = values[i]
    values_model += values_model[:1]  # Close the loop
    ax.plot(angles, values_model, linewidth=2, label=model)
    ax.fill(angles, values_model, alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Model Comparison Across Multiple Dimensions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_results/radar_comparison.png', dpi=300)
plt.close()

# Figure 7: Transfer model training history
epochs = list(range(1, 11))
transfer_train_acc = [49.51, 51.57, 49.34, 50.00, 52.09, 50.03, 51.86, 52.57, 52.29, 50.89]
transfer_val_acc = [48.80, 50.67, 51.07, 48.80, 51.47, 50.13, 51.20, 49.33, 50.40, 49.60]
transfer_train_loss = [0.7017, 0.6939, 0.6968, 0.6954, 0.6928, 0.6935, 0.6913, 0.6929, 0.6921, 0.6922]
transfer_val_loss = [0.6950, 0.6930, 0.6929, 0.6937, 0.6931, 0.6941, 0.6942, 0.6942, 0.6953, 0.6944]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Accuracy plot
ax1.plot(epochs, transfer_train_acc, 'o-', label='Training Accuracy', color='#3498db')
ax1.plot(epochs, transfer_val_acc, 'o-', label='Validation Accuracy', color='#e74c3c')
ax1.set_title('Transfer Model Training and Validation Accuracy', fontsize=16, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, linestyle='--', alpha=0.7)

# Loss plot
ax2.plot(epochs, transfer_train_loss, 'o-', label='Training Loss', color='#3498db')
ax2.plot(epochs, transfer_val_loss, 'o-', label='Validation Loss', color='#e74c3c')
ax2.set_title('Transfer Model Training and Validation Loss', fontsize=16, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('visualization_results/transfer_model_history.png', dpi=300)
plt.close()

# Figure 8: Model architecture comparison
fig, ax = plt.subplots(figsize=(12, 8))

# Define architecture details for each model
architectures = {
    'Hybrid Model': ['Feature Extraction', '24D Input Vector', 'Dense (64)', 'Dropout (0.3)', 'Dense (32)', 'Dropout (0.3)', 'Output (1)'],
    'CNN Model': ['Image Input', 'Conv2D (32)', 'MaxPooling2D', 'Conv2D (64)', 'MaxPooling2D', 'Conv2D (128)', 'MaxPooling2D', 'Flatten', 'Dense (512)', 'Dropout (0.5)', 'Output (1)'],
    'Transfer Model': ['24D Input Vector', 'Dense (64)', 'Dropout (0.3)', 'Dense (32)', 'Dropout (0.3)', 'Output (1)']
}

# Set positions for each model
positions = [0, 1, 2]
colors = ['#3498db', '#2ecc71', '#e74c3c']

# Draw the architecture boxes
for i, model in enumerate(models):
    layers = architectures[model]
    for j, layer in enumerate(layers):
        y = i
        x = j / (len(layers) - 1) * 10  # Scale to fit
        rect = plt.Rectangle((x, y-0.3), 0.8, 0.6, alpha=0.7, color=colors[i])
        ax.add_patch(rect)
        ax.text(x+0.4, y, layer, ha='center', va='center', fontweight='bold')

# Set plot boundaries and labels
ax.set_xlim(-1, 11)
ax.set_ylim(-0.5, 2.5)
ax.set_yticks(positions)
ax.set_yticklabels(models)
ax.set_xticks([])
ax.set_title('Model Architecture Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_results/architecture_comparison.png', dpi=300)
plt.close()

print("Visualization images have been generated in the 'visualization_results' directory.") 