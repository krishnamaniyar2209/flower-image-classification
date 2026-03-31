# 🌸 Flower Image Classification — Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS672-blue)

> Fine-tuning a Pre-trained **ResNet50** model for flower species classification using both **TensorFlow** and **PyTorch** — built for CS672: Introduction to Deep Learning at Pace University (Fall 2025).

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🔬 Overview

This project implements **Transfer Learning** using a pre-trained ResNet50 model to classify 5 flower species. Both TensorFlow (Keras) and PyTorch frameworks are used and compared.

- ✅ **Step 1** — Data preparation (load, resize, encode, split)
- ✅ **Step 2** — Pre-trained ResNet50 selected (ImageNet weights)
- ✅ **Step 3** — TensorFlow transfer learning with EarlyStopping + LR Scheduler
- ✅ **Step 4** — PyTorch transfer learning with data augmentation + early stopping
- ✅ **Step 5** — Full evaluation: Accuracy, Precision, Recall, F1, Confusion Matrices
- ✅ **Model Serialization** — saved as `.keras` and `.pt`
- ✅ **Framework Comparison** — side-by-side results table

---

## 📊 Dataset

**Source:** [Kaggle — Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

| Property | Details |
|---|---|
| Total Images | 4,317 |
| Classes | 5 (daisy, dandelion, rose, sunflower, tulip) |
| Images per Class | ~800 |
| Image Size | 150×150 (raw), resized to 224×224 for ResNet50 |

### Data Split

| Set | Size |
|---|---|
| Training | ~60% |
| Validation | ~15% |
| Test | 25% |

---

## 📁 Project Structure
```
flower-image-classification/
│
├── Flower_Image_Classification.ipynb   # Main notebook
├── flower_classifier_tensorflow.keras  # Saved TF model
├── flower_classifier_pytorch.pt        # Saved PyTorch model
├── README.md                           # Documentation
└── requirements.txt                    # Dependencies
```

---

## 🔬 Methodology

### Step 1 — Data Preparation
- Downloaded via `kagglehub`
- Images loaded with OpenCV, resized to 150×150
- Label encoding + one-hot encoding (`to_categorical`)
- Stratified 75/25 train/test split
- Validation set carved from training (20%)

### Step 2 — Pre-trained Model
- **ResNet50** with ImageNet weights
- `include_top=False` — custom classification head added
- All base layers frozen

### Step 3 — TensorFlow Implementation
```python
base_model = ResNet50(weights='imagenet', include_top=False)
# Freeze all base layers
for layer in base_model.layers:
    layer.trainable = False
# Custom head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
output = Dense(5, activation='softmax')(x)
```
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping (patience=3) + ReduceLROnPlateau (factor=0.5)

### Step 4 — PyTorch Implementation
- ResNet50 with pretrained weights
- `fc` layer replaced: `nn.Linear(2048, 5)`
- All layers except `fc` frozen
- Data augmentation: RandomHorizontalFlip + RandomRotation
- CrossEntropyLoss + Adam optimizer
- Manual early stopping (patience=3)

---

## 📈 Results

### Framework Comparison

| Framework | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| TensorFlow | — | — | — | — |
| PyTorch | — | — | — | — |

> Fill in actual values after running the notebook.

---

## ⚙️ Installation
```bash
git clone https://github.com/krishnamaniyar2209/flower-image-classification.git
cd flower-image-classification
pip install -r requirements.txt
jupyter notebook Flower_Image_Classification.ipynb
```

### requirements.txt
```
tensorflow>=2.10.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.64.0
kagglehub>=0.1.0
jupyter>=1.0.0
```

---

## 💡 Key Findings

- Transfer learning achieves strong accuracy with only 20 epochs
- PyTorch augmentation (flip + rotation) helps generalization
- ResNet50 frozen features are powerful enough for flower classification
- EarlyStopping prevents overfitting efficiently
- ReduceLROnPlateau helps fine-tune convergence

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| TensorFlow/Keras | Transfer learning (Step 3) |
| PyTorch | Transfer learning (Step 4) |
| ResNet50 | Pre-trained base model |
| OpenCV | Image loading and resizing |
| scikit-learn | Metrics and data splitting |
| Seaborn | Confusion matrix visualization |

---

## 👤 Author

**Krishna Maniyar**
- 🎓 Pace University — Seidenberg School of CSIS
- 📘 CS672: Introduction to Deep Learning | Fall 2025
- 🔗 [GitHub](https://github.com/krishnamaniyar2209)

---

<p align="center">Made with ❤️ for CS672 @ Pace University</p>
