# Sentiment Analysis & Emotion Classifier

This project provides a deep learning-based solution for emotion classification in text data using Python, Pandas, Scikit-learn, and TensorFlow/Keras. The model classifies input comments into three emotion categories: **anger**, **joy**, and **fear**.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

## Overview

This notebook demonstrates the end-to-end pipeline for emotion classification:
- Data loading and exploration
- Preprocessing and label encoding
- Train-test split
- Tokenization and sequence padding
- Deep learning model definition (Embedding + LSTM)
- Model training and evaluation
- Visualization of training progress and confusion matrix
- Example predictions

## Dataset

The dataset is expected in CSV format, named `Emotion_classify_Data.csv`, with the following columns:
- `Comment`: The text of the comment or sentence.
- `Emotion`: The corresponding emotion label (`anger`, `joy`, or `fear`).

## Model Architecture

- **Embedding Layer**: Converts words to dense vector representations.
- **LSTM Layer**: Captures sequence information from text.
- **Dense Layer with Softmax**: Outputs probability distribution over emotion classes.

**Summary:**
- Input: Padded sequences of tokens (max length 100, vocab size 10,000)
- Embedding dimension: 16
- LSTM units: 64
- Output: 3 classes (anger, joy, fear)

## Usage

1. **Clone the repository** and add your `Emotion_classify_Data.csv` into the working directory.

2. **Run the notebook**:  
   Open `Sentiment_Analysis_Emotion_Classifier.ipynb` in Jupyter or Google Colab.

3. **Step through the notebook** to:
   - Load and explore the data
   - Train and evaluate the LSTM-based classifier
   - Visualize results

No additional configuration is required.

## Results

- **Accuracy**: The model achieves high accuracy (~93%) on a held-out test set.
- **Classification Report**: Precision, recall, and F1-score are reported for each emotion class.
- **Confusion Matrix**: A heatmap visualizes prediction errors.
- **Sample Predictions**: The notebook displays actual vs. predicted emotions for several test examples.

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow (>=2.x)

Install requirements with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Acknowledgements

- Inspired by open-source work on sentiment analysis and NLP with Keras/TensorFlow.
- Dataset and task adapted for educational/demo purposes.

---

**Author**: [Your Name or GitHub handle]  
**License**: MIT (if desired)
