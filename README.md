# ECG Classification with CNN using Spectrograms

A deep learning project for classifying ECG (Electrocardiogram) signals into different arrhythmia types using Convolutional Neural Networks (CNN) on spectrogram representations.

## 📋 Overview

This project implements a 2D CNN model to classify ECG signals from the MIT-BIH Arrhythmia Database. The approach converts 1D ECG signals into 2D spectrograms (time-frequency representations) and uses a state-of-the-art CNN architecture for classification.

## 🎯 Features

- **Data Analysis**: Comprehensive analysis of ECG signal distributions across different classes
- **Imbalanced Data Handling**: Uses SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset
- **Spectrogram Conversion**: Transforms 1D ECG signals into 2D spectrograms for better feature extraction
- **2D CNN Architecture**: Deep convolutional neural network with modern techniques:
  - Batch Normalization
  - Dropout regularization
  - AdamW optimizer with weight decay
  - Global Average Pooling
- **Visualization**: 
  - Class distribution analysis
  - Sample ECG signals visualization
  - Spectrogram visualization for each class
  - Training history plots
  - Confusion matrix
- **Comprehensive Metrics**: Reports training, validation, and test accuracy with per-class performance

## 📊 Dataset

- **Source**: MIT-BIH Arrhythmia Database
- **Format**: Pre-split into training and test sets
- **Classes**: 5 different ECG signal types (arrhythmia classifications)
- **Note**: Dataset files are not included in this repository due to size constraints

## 🏗️ Architecture

The model uses a 2D CNN architecture optimized for spectrogram classification:

- **Input**: Spectrograms (frequency × time × 1 channel)
- **Convolutional Blocks**: 4 blocks with increasing filters (32 → 64 → 128 → 256)
- **Regularization**: Batch Normalization and Dropout at each block
- **Pooling**: MaxPooling2D for dimensionality reduction
- **Classification Head**: Global Average Pooling → Dense layers → Softmax output

## 🚀 Usage

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy tensorflow
```

### Data Setup

Place your MIT-BIH dataset files in the appropriate location:
- Training: `/kaggle/input/heartbeat/mitbih_train.csv`
- Test: `/kaggle/input/heartbeat/mitbih_test.csv`

Or modify the file paths in the notebook to match your local setup.

### Running the Notebook

1. Open `ecg_classification_with_CNN.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. The notebook will:
   - Load and analyze the data
   - Apply SMOTE for balancing
   - Convert signals to spectrograms
   - Train the CNN model for 10 epochs
   - Evaluate and display results

## 📈 Results

The model provides:
- Training and validation loss/accuracy curves
- Test set performance metrics
- Per-class accuracy breakdown
- Confusion matrix visualization
- Classification report with precision, recall, and F1-scores

## 🔬 Methodology

1. **Data Preprocessing**:
   - Standardization using StandardScaler
   - Stratified train-test split (already provided in dataset)
   - SMOTE oversampling on training set only

2. **Feature Engineering**:
   - Convert 1D ECG signals to 2D spectrograms using Short-Time Fourier Transform (STFT)
   - Normalize spectrograms to [0, 1] range
   - dB scale conversion for better visualization

3. **Model Training**:
   - 2D CNN with state-of-the-art optimizations
   - AdamW optimizer with learning rate scheduling
   - Early stopping and learning rate reduction callbacks
   - 10 epochs training with validation monitoring

## 📁 Project Structure

```
.
├── README.md                          # Project documentation
├── ecg_classification_with_CNN.ipynb  # Main implementation notebook
└── .gitignore                         # Git ignore file
```

## 🛠️ Technologies

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **imbalanced-learn**: SMOTE implementation
- **scipy**: Signal processing (spectrogram generation)
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization

## 📝 Notes

- The dataset is imbalanced, which is why SMOTE is applied
- Spectrograms provide better frequency-domain features than raw signals
- The model uses modern deep learning best practices (BatchNorm, Dropout, AdamW)
- CSV files are excluded from version control due to size

## 📄 License

This project is for academic/research purposes.

## 👤 Authors

Egehan Eralp
Aslı Atabek

## 🙏 Acknowledgments

- MIT-BIH Arrhythmia Database for providing the dataset
- Kaggle community for dataset hosting

