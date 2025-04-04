# Image Classification with Transfer Learning and SVM

This repository contains code for image classification using Transfer Learning with DenseNet models (DenseNet121, DenseNet169, DenseNet201) and Support Vector Machines (SVM) with different kernel functions.

## Overview

This project implements a comprehensive image classification pipeline that:

1. Extracts features from images using pre-trained DenseNet models
2. Trains SVM classifiers with different kernel functions on the extracted features
3. Evaluates performance using multiple metrics
4. Visualizes results with confusion matrices and ROC curves

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm

Install the required packages using:

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn tqdm
```

## Dataset Structure

Your dataset should be organized in the following directory structure:

```
dataset_folder/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

Where each subdirectory represents a different class.

## Usage

### Google Colab

1. Upload the code to Google Colab
2. Upload your dataset to Google Drive
3. Update the `DATA_DIR` path in the `Config` class to point to your dataset location
4. Run the notebook with GPU acceleration enabled

### Local Execution

1. Clone this repository
2. Update the `DATA_DIR` path in the `Config` class to point to your dataset location
3. Run the main script:

```bash
python image_classification.py
```

## Features

### Transfer Learning Models

The code uses three DenseNet variants:
- **DenseNet121**: Lighter model with 8 million parameters
- **DenseNet169**: Medium model with 14 million parameters
- **DenseNet201**: Deeper model with 20 million parameters

### SVM Kernels

Four different SVM kernel functions are used:
- **Linear**: Uses a linear decision boundary
- **RBF (Radial Basis Function)**: Creates non-linear decision boundaries using Gaussian functions
- **Sigmoid**: Creates decision boundaries using sigmoid function
- **Polynomial**: Creates curved decision boundaries using polynomial functions

### Performance Metrics

For each model and kernel combination, the code calculates:
- **Accuracy**: Overall correctness
- **Precision**: Ability to avoid false positives
- **Recall**: Ability to find all positive samples
- **F1 Score**: Harmonic mean of precision and recall
- **F2 Score**: Similar to F1 but weighs recall higher than precision
- **MCC (Matthews Correlation Coefficient)**: Correlation between predicted and actual values
- **Kappa Score**: Measures agreement level compared to random chance
- **ROC AUC**: Area under the Receiver Operating Characteristic curve

### Visualizations

- **Confusion Matrices**: Shows which classes are confused with each other
- **ROC Curves**: Shows the trade-off between true positive rate and false positive rate
- **Comparison Plots**: Bar charts comparing all metrics across models and kernels

## Output

The code generates:
1. A CSV file with all performance metrics (`model_comparison_results.csv`)
2. Confusion matrices for each model and kernel combination
3. ROC curves for each model and kernel combination
4. Comparison bar charts (`metrics_comparison.png`)

## Pipeline Description

1. **Data Loading and Preprocessing**:
   - Images are loaded and preprocessed using `ImageDataGenerator`
   - Training data is augmented with rotations, shifts, flips, etc.
   - Images are resized to 224×224 pixels (standard for DenseNet)
   - Pixel values are scaled to [0,1]

2. **Feature Extraction**:
   - Pre-trained DenseNet models are used as feature extractors
   - The top classification layer is removed
   - A Global Average Pooling layer is added to get feature vectors
   - Features are extracted for all images

3. **SVM Classification**:
   - Features are standardized using `StandardScaler`
   - SVM models are trained with different kernels
   - Models are evaluated on validation data

4. **Evaluation and Visualization**:
   - Performance metrics are calculated
   - Confusion matrices and ROC curves are plotted
   - Comparison plots are generated

## Customization

You can customize the pipeline by modifying:
- **Image size**: Change `IMG_SIZE` in the `Config` class
- **Batch size**: Change `BATCH_SIZE` in the `Config` class
- **Models**: Add or remove models in the `MODELS` dictionary in the `Config` class
- **SVM kernels**: Add or remove kernels in the `SVM_KERNELS` list in the `Config` class
- **Data augmentation**: Adjust parameters in the `train_datagen` in the `create_data_generators` function

## Troubleshooting

- **Memory issues**: Reduce batch size or image size
- **Slow processing**: Enable GPU acceleration in Google Colab
- **Poor performance**: Try different models or kernels, or add more data augmentation

## License

[MIT License](LICENSE)

## Acknowledgments

- The DenseNet implementation is based on the TensorFlow/Keras Applications module
- The evaluation metrics implementation uses scikit-learn
