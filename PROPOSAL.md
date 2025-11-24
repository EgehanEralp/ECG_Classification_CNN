# Project Proposal: ECG Classification with CNN using Spectrograms

## Project Title

**ECG Arrhythmia Classification Using Deep Learning: A Spectrogram-Based Convolutional Neural Network Approach**

## Aim/Motivation of this Project

### Motivation

Cardiovascular diseases are among the leading causes of death worldwide, with arrhythmias being a significant contributor. Early and accurate detection of cardiac arrhythmias is crucial for timely medical intervention and improved patient outcomes. Traditional manual ECG analysis is time-consuming, subjective, and requires extensive expertise, making it challenging to scale for large populations.

### Aim

The primary aim of this project is to develop an automated, accurate, and efficient deep learning system for classifying ECG signals into different arrhythmia types. Specifically, this project aims to:

1. **Automate ECG Classification**: Develop a deep learning model that can automatically classify ECG signals into multiple arrhythmia categories without manual intervention.

2. **Improve Classification Accuracy**: Leverage advanced signal processing techniques (spectrograms) combined with state-of-the-art CNN architectures to achieve high classification accuracy.

3. **Handle Imbalanced Data**: Address the challenge of imbalanced datasets commonly found in medical data through appropriate data augmentation and balancing techniques.

4. **Provide Interpretable Results**: Generate comprehensive visualizations and metrics to understand model performance and class-specific accuracy.

5. **Demonstrate Scalability**: Show that the proposed approach can be applied to large-scale ECG datasets efficiently.

### Significance

This project contributes to the field of automated medical diagnosis by:
- Reducing the burden on healthcare professionals
- Enabling faster diagnosis and treatment decisions
- Potentially improving patient outcomes through early detection
- Providing a foundation for real-time ECG monitoring systems

## Datasets

### Dataset: MIT-BIH Arrhythmia Database

**Source**: PhysioNet - MIT-BIH Arrhythmia Database

**Description**: 
The MIT-BIH Arrhythmia Database is a widely recognized benchmark dataset for ECG signal analysis. It contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory.

**Dataset Characteristics**:
- **Total Samples**: ~100,000+ ECG signal segments
- **Number of Classes**: 5 distinct arrhythmia types
  - Normal (N): Normal sinus rhythm
  - Ventricular Ectopic Beat (VEB): Premature ventricular contractions
  - Supraventricular Ectopic Beat (SVEB): Premature atrial contractions
  - Fusion (F): Fusion of normal and ventricular beats
  - Unknown (Q): Unclassifiable beats

**Data Format**:
- Pre-processed ECG signal segments
- Each sample contains temporal features representing ECG waveform characteristics
- Pre-split into training and test sets for evaluation

**Data Challenges**:
- **Class Imbalance**: Significant imbalance between classes (e.g., Normal class dominates with ~90% of samples)
- **Signal Variability**: High variability in signal morphology across different subjects and conditions
- **Noise**: Real-world ECG signals may contain artifacts and noise

**Data Access**:
- Available through Kaggle: `/kaggle/input/heartbeat/mitbih_train.csv` and `mitbih_test.csv`
- Standardized format suitable for machine learning applications

## High-level Solution Pipeline

### Overview

The solution pipeline transforms raw ECG signals into spectrogram representations and employs a 2D Convolutional Neural Network for classification. The pipeline consists of five main stages:

```
Raw ECG Signals → Data Preprocessing → Spectrogram Conversion → CNN Training → Classification & Evaluation
```

### Detailed Pipeline Stages

#### Stage 1: Data Loading and Analysis
- **Input**: CSV files containing ECG signal features and labels
- **Process**: 
  - Load training and test datasets
  - Analyze class distribution and data characteristics
  - Identify data quality issues and imbalances
- **Output**: Analyzed dataset with class distribution statistics

#### Stage 2: Data Preprocessing and Balancing
- **Input**: Raw ECG signal features
- **Process**:
  - **Standardization**: Normalize features using StandardScaler to ensure consistent scale
  - **Stratified Splitting**: Maintain class proportions (already provided in dataset)
  - **SMOTE Oversampling**: Apply Synthetic Minority Oversampling Technique to balance the training set
    - Generate synthetic samples for minority classes
    - Preserve class relationships while increasing sample diversity
- **Output**: Balanced and standardized training dataset

#### Stage 3: Spectrogram Conversion
- **Input**: 1D ECG signal arrays
- **Process**:
  - **Short-Time Fourier Transform (STFT)**: Convert time-domain signals to time-frequency domain
    - Parameters: Sampling frequency (360 Hz), window size (64 samples), overlap (32 samples)
  - **Magnitude Calculation**: Compute power spectral density
  - **dB Conversion**: Transform to logarithmic scale (decibels) for better dynamic range
  - **Normalization**: Scale spectrograms to [0, 1] range for neural network input
- **Output**: 2D spectrogram images (frequency × time × 1 channel)

#### Stage 4: CNN Model Architecture and Training
- **Input**: Spectrogram images
- **Model Architecture**:
  - **Convolutional Layers**: 4 blocks with increasing filter depth (32 → 64 → 128 → 256)
  - **Regularization**: 
    - Batch Normalization after each convolutional layer
    - Dropout layers (0.25-0.5) to prevent overfitting
  - **Pooling**: MaxPooling2D for dimensionality reduction
  - **Classification Head**: 
    - Global Average Pooling
    - Dense layers (512 → 256 → num_classes)
    - Softmax activation for multi-class classification
- **Training Configuration**:
  - **Optimizer**: AdamW with learning rate 0.001 and weight decay 0.01
  - **Loss Function**: Categorical cross-entropy
  - **Training**: 10 epochs with validation split (20%)
  - **Callbacks**: Early stopping and learning rate reduction
  - **Batch Size**: 128
- **Output**: Trained CNN model with learned feature representations

#### Stage 5: Evaluation and Visualization
- **Input**: Trained model and test spectrograms
- **Process**:
  - Generate predictions on test set
  - Calculate classification metrics
  - Visualize results
- **Output**:
  - Training/validation loss and accuracy curves
  - Test set performance metrics
  - Confusion matrix
  - Per-class accuracy breakdown
  - Classification report (precision, recall, F1-score)

### Key Technical Decisions

1. **Spectrogram Approach**: Chosen over raw signals because:
   - Captures both temporal and frequency domain features
   - Better suited for 2D CNN architectures
   - Reveals patterns not easily visible in time domain

2. **SMOTE for Balancing**: Selected because:
   - Preserves original data distribution characteristics
   - More effective than simple oversampling
   - Prevents model bias toward majority class

3. **2D CNN Architecture**: Optimal for:
   - Processing image-like spectrogram data
   - Learning hierarchical spatial-frequency features
   - Leveraging proven image classification techniques

## Expected Output

### Primary Outputs

1. **Trained CNN Model**
   - Fully trained 2D CNN model capable of classifying ECG signals
   - Model weights and architecture saved for deployment
   - Capable of real-time inference on new ECG signals

2. **Performance Metrics**
   - **Overall Accuracy**: Expected > 95% on test set
   - **Per-Class Accuracy**: Detailed breakdown for each arrhythmia type
   - **Confusion Matrix**: Visual representation of classification performance
   - **Classification Report**: Precision, recall, and F1-scores for each class

3. **Visualization Results**
   - Class distribution plots showing dataset characteristics
   - Sample ECG signals and their corresponding spectrograms
   - Training history curves (loss and accuracy over epochs)
   - Confusion matrix heatmap
   - Per-class performance comparisons

4. **Analysis and Insights**
   - Identification of which classes are most/least accurately classified
   - Understanding of model strengths and limitations
   - Recommendations for model improvement

### Expected Performance

- **Training Accuracy**: > 98% (after SMOTE balancing)
- **Validation Accuracy**: > 95% (indicating good generalization)
- **Test Accuracy**: > 95% (on unseen data)
- **Per-Class Performance**: 
  - Normal (N): > 98% (majority class)
  - VEB: > 90% (well-represented minority class)
  - SVEB: > 85% (minority class)
  - F: > 80% (rare class)
  - Q: Variable (very rare class)

### Deliverables

1. **Jupyter Notebook** (`ecg_classification_with_CNN.ipynb`)
   - Complete implementation with all code cells
   - Documentation and comments
   - Reproducible results

2. **Documentation**
   - README.md: User guide and project overview
   - PROPOSAL.md: This project proposal document

3. **Results Summary**
   - Final model performance metrics
   - Key findings and observations
   - Potential improvements and future work

### Potential Applications

- **Clinical Decision Support**: Assist healthcare professionals in ECG interpretation
- **Telemedicine**: Enable remote ECG monitoring and analysis
- **Screening Programs**: Large-scale population health screening
- **Research**: Foundation for further arrhythmia research
- **Educational**: Teaching tool for medical students and residents

### Limitations and Future Work

- **Dataset Limitations**: Single database, may not generalize to all populations
- **Real-time Processing**: Current implementation optimized for batch processing
- **Interpretability**: Could benefit from attention mechanisms or explainable AI techniques
- **Multi-lead ECG**: Current approach uses single-lead; could extend to multi-lead analysis
- **Clinical Validation**: Requires validation on real-world clinical data

---

**Authors**: Egehan Eralp, Aslı Atabek

**Institution**: Koç University

**Date**: 2025

