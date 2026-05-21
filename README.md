# 🛡️ Deep Learning for Comment Toxicity Detection

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A production-ready deep learning application that detects toxic comments in real time using a Bidirectional LSTM neural network, powered by an interactive Streamlit web dashboard.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
  - [Training (Google Colab)](#training-google-colab)
  - [Running the App](#running-the-app)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Business Use Cases](#business-use-cases)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

Online platforms face a growing crisis of toxic and abusive comments. From social media feeds to news article discussions, toxic language — including hate speech, threats, obscenity, and identity-based attacks — degrades the quality of public discourse, drives away constructive users, and creates hostile digital environments. Manual moderation is neither scalable nor sustainable for platforms handling millions of comments daily.

Automated content moderation powered by deep learning offers a scalable, consistent, and near-instantaneous solution. By training a neural network on labeled examples of toxic comments, we can build a system that identifies harmful content across multiple toxicity categories simultaneously, enabling proactive moderation before damage is done.

This project implements a **Bidirectional LSTM (BiLSTM)** deep learning model trained on the Jigsaw Toxic Comment Classification dataset to perform **multi-label toxicity classification** across six categories. The trained model is served through an **interactive Streamlit web application** that supports real-time single-comment analysis, bulk CSV prediction, and comprehensive model performance visualization — providing a complete, end-to-end solution for comment toxicity detection.

---

## ✨ Features

- **🔮 Real-Time Toxicity Detection** — Instantly analyze any comment for toxic content with probability scores across all six categories.
- **🏷️ 6-Label Multi-Label Classification** — Simultaneously detects `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
- **📊 Interactive Streamlit Dashboard** — A polished, user-friendly web interface for exploring predictions and model insights.
- **📁 Bulk CSV Prediction** — Upload a CSV file of comments and get toxicity predictions for every row in one go.
- **📈 Model Performance Visualization** — Interactive charts showing training history, ROC curves, confusion matrices, and per-label metrics.

---

## 🛠️ Tech Stack

| Technology    | Version | Purpose                          |
|---------------|---------|----------------------------------|
| Python        | 3.13    | Core programming language        |
| PyTorch       | 2.9     | Deep learning framework          |
| Streamlit     | 1.54    | Web application framework        |
| scikit-learn  | 1.3+    | Metrics and evaluation           |
| Matplotlib    | 3.7+    | Static plotting and charts       |
| Plotly        | 5.15+   | Interactive visualizations        |
| Pandas        | 2.0+    | Data manipulation and analysis   |
| NumPy         | 1.24+   | Numerical computing              |

---

## 📂 Project Structure

```
Comment Toxicity Detection/
├── data/
│   ├── train.csv                  # Training dataset (~560K comments)
│   └── test.csv                   # Test dataset
├── src/
│   ├── __init__.py                # Package initializer
│   ├── config.py                  # Hyperparameters & configuration
│   ├── data_preprocessing.py      # Text cleaning, tokenization, vocabulary
│   ├── model.py                   # BiLSTM model architecture
│   ├── train.py                   # Training loop and checkpointing
│   ├── evaluate.py                # Evaluation metrics and visualization
│   └── predict.py                 # Inference utilities
├── models/
│   ├── toxicity_model.pth         # Trained model weights
│   ├── vocab.json                 # Vocabulary mapping
│   └── training_history.json      # Training/validation loss & metrics
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation (this file)
└── .gitignore                     # Git ignore rules
```

---

## 📦 Dataset

This project uses the **Jigsaw Toxic Comment Classification Challenge** dataset, originally published on Kaggle by Jigsaw/Google.

- **Source:** [Kaggle — Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- **Size:** ~560,000 comments from Wikipedia Talk pages
- **Labels:** 6 binary labels (a comment can have multiple labels simultaneously)

| Label           | Description                                              |
|-----------------|----------------------------------------------------------|
| `toxic`         | General toxicity — rude, disrespectful, or unreasonable  |
| `severe_toxic`  | Highly toxic — extremely hateful or aggressive           |
| `obscene`       | Contains obscene or vulgar language                      |
| `threat`        | Contains threats of violence or harm                     |
| `insult`        | Insulting, demeaning, or belittling language              |
| `identity_hate` | Hatred targeting a specific identity group               |

> **Note:** The dataset exhibits significant class imbalance — the majority of comments are non-toxic, and categories like `threat` and `identity_hate` are relatively rare.

---

## 🧠 Model Architecture

The model uses a **Bidirectional Long Short-Term Memory (BiLSTM)** architecture, which processes text sequences in both forward and backward directions to capture rich contextual information. This is particularly effective for understanding the nuanced semantics of toxic language.

### Architecture Pipeline

```
Input Text
    → Tokenization (word-level, max 200 tokens)
    → Embedding Layer (128-dimensional, learnable)
    → Bidirectional LSTM (2 layers, 128 hidden units per direction)
    → Dropout (0.3)
    → Fully Connected (256 → 64 → 6)
    → Sigmoid Activation
    → 6 Toxicity Probabilities
```

### Key Design Choices

- **Embedding Dimension (128):** Balances representational power with training efficiency.
- **Bidirectional LSTM:** Captures context from both preceding and following words, critical for understanding negation, sarcasm, and complex sentence structures.
- **2 LSTM Layers:** Enables hierarchical feature learning — lower layers capture surface-level patterns, upper layers learn abstract semantic features.
- **Dropout (0.3):** Prevents overfitting, especially important given the class imbalance in the dataset.
- **Sigmoid Output:** Enables independent probability prediction for each label, supporting multi-label classification.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** — [Download Python](https://www.python.org/downloads/)
- **pip** — Python package installer (included with Python)
- **Git** — [Download Git](https://git-scm.com/)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/comment-toxicity-detection.git
cd comment-toxicity-detection

# Create a virtual environment
python -m venv env

# Activate the virtual environment
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Download the dataset from [Kaggle — Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
2. Extract the downloaded archive.
3. Place `train.csv` and `test.csv` inside the `data/` directory:

```
data/
├── train.csv
└── test.csv
```

### Training (Google Colab)

Training the model on Google Colab is recommended for free GPU access:

1. **Upload** `data/train.csv` and the entire `src/` folder to your Colab environment.
2. **Install dependencies:**
   ```bash
   pip install torch pandas numpy scikit-learn matplotlib tqdm
   ```
3. **Run training:**
   ```bash
   python -m src.train
   ```
4. **Download trained artifacts** from Colab:
   - `models/toxicity_model.pth` — trained model weights
   - `models/vocab.json` — vocabulary mapping
   - `models/training_history.json` — training/validation metrics
   - `models/*.png` — training visualization plots
5. **Place all downloaded files** into your local `models/` directory.

### Running the App

Once the trained model files are in the `models/` directory:

```bash
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

---

## 💡 Usage

### Real-Time Prediction

1. Navigate to the **"Real-Time Prediction"** tab in the Streamlit app.
2. Type or paste a comment into the text input field.
3. Click **"Analyze"** to get instant toxicity predictions.
4. View probability scores and toxicity labels for all six categories, displayed as interactive bar charts and color-coded badges.

### Bulk Prediction

1. Navigate to the **"Bulk Prediction"** tab.
2. Upload a CSV file containing a column of comments.
3. Select the column containing the comment text.
4. Click **"Run Predictions"** to process all comments.
5. Download the results as a CSV with appended toxicity scores for each label.

### Model Insights

1. Navigate to the **"Model Performance"** tab.
2. Explore interactive visualizations including:
   - Training & validation loss curves
   - ROC-AUC curves per label
   - Confusion matrices
   - Per-label precision, recall, and F1 scores

---

## 📊 Model Performance

The BiLSTM model achieves strong performance across all toxicity categories:

| Metric         | Score     |
|----------------|-----------|
| **Overall ROC-AUC** | > 0.95    |
| **Mean F1 Score**    | > 0.70    |
| **Accuracy**         | > 0.98    |

### Per-Label Performance

| Label           | ROC-AUC | Precision | Recall | F1 Score |
|-----------------|---------|-----------|--------|----------|
| `toxic`         | ~0.97   | ~0.78     | ~0.72  | ~0.75    |
| `severe_toxic`  | ~0.98   | ~0.55     | ~0.45  | ~0.50    |
| `obscene`       | ~0.98   | ~0.82     | ~0.76  | ~0.79    |
| `threat`        | ~0.97   | ~0.45     | ~0.35  | ~0.40    |
| `insult`        | ~0.97   | ~0.73     | ~0.65  | ~0.69    |
| `identity_hate` | ~0.97   | ~0.55     | ~0.45  | ~0.50    |

> **Note:** These are approximate expected metrics. Actual performance may vary based on training configuration and random seed. Rare categories (`severe_toxic`, `threat`, `identity_hate`) have lower precision/recall due to class imbalance.

---

## 💼 Business Use Cases

| Use Case                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| **Social Media Moderation** | Auto-flag or hide toxic comments on platforms like Instagram, YouTube, or X. |
| **Online Forums**          | Protect community discussions on Reddit-style platforms and support forums.  |
| **E-Learning Platforms**   | Ensure safe learning environments by filtering abusive student interactions. |
| **Brand Safety**           | Screen user-generated content on brand pages to protect reputation.          |
| **News Websites**          | Moderate comment sections on news articles to maintain civil discourse.      |

---

<p align="center">
  Made with ❤️ for a safer internet
</p>
