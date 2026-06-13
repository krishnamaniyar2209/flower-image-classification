# 🌸 Flower Image Classification — Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS672-blue)

> Fine-tuning a pre-trained **ResNet50** for 5-class flower classification in **both TensorFlow and PyTorch**, then comparing the two frameworks — built for CS672: Introduction to Deep Learning at Pace University (Fall 2025).

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

This project implements **transfer learning** with a pre-trained ResNet50 (ImageNet weights) to classify 5 flower species, implemented and benchmarked in **both TensorFlow (Keras) and PyTorch**.

- ✅ Data preparation (load, resize, encode, stratified split)
- ✅ Pre-trained ResNet50 with frozen base + custom head
- ✅ TensorFlow training with EarlyStopping + ReduceLROnPlateau
- ✅ PyTorch training with data augmentation + early stopping
- ✅ Full evaluation: Accuracy, Precision, Recall, F1, confusion matrices
- ✅ Side-by-side framework comparison

---

## 📊 Dataset

**Source:** [Kaggle — Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

| Property | Details |
|---|---|
| Total Images | 4,317 |
| Classes | 5 (daisy, dandelion, rose, sunflower, tulip) |
| Per-class counts | dandelion 1,052 · tulip 984 · rose 784 · daisy 764 · sunflower 733 |
| Image Size | 150×150 (raw), resized to 224×224 for ResNet50 |

### Data Split
| Set | Size | Share |
|---|---|---|
| Training | 2,589 | 60% |
| Validation | 648 | 15% |
| Test | 1,080 | 25% |

---

## 📁 Project Structure
```
flower-image-classification/
│
├── Flower_Image_Classification.ipynb   # Main notebook
├── README.md                           # Documentation
└── requirements.txt                    # Dependencies

# Generated locally (not committed — large model files):
#   flower_classifier_tensorflow.keras
#   flower_classifier_pytorch.pt
```

---

## 🔬 Methodology

### Step 1 — Data Preparation
- Downloaded via `kagglehub`; images loaded with OpenCV, resized to 150×150
- Label + one-hot encoding (`to_categorical`)
- Stratified 75/25 train/test split; validation (20%) carved from training

### Step 2 — Pre-trained Model
- **ResNet50** with ImageNet weights, `include_top=False`, all base layers frozen, custom head added

### Step 3 — TensorFlow Implementation
```python
base_model = ResNet50(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
output = Dense(5, activation='softmax')(x)
```
- Adam (lr=0.001), Categorical Crossentropy
- Callbacks: EarlyStopping (patience=3) + ReduceLROnPlateau (factor=0.5)

### Step 4 — PyTorch Implementation
- ResNet50 pretrained; `fc` replaced with `nn.Linear(2048, 5)`, all other layers frozen
- Augmentation: RandomHorizontalFlip + RandomRotation
- CrossEntropyLoss + Adam, manual early stopping (patience=3)

---

## 📈 Results

### Framework Comparison
| Framework | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| TensorFlow | 0.860 | 0.862 | 0.854 | 0.857 |
| **PyTorch** | **0.881** | **0.883** | **0.878** | **0.880** |

> **PyTorch outperformed TensorFlow** by ~2 points across all metrics — likely due to its added data augmentation (flip + rotation). Both models converged within a few epochs thanks to frozen ResNet50 features.

### Per-Class Performance (PyTorch — best model)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| daisy | 0.89 | 0.87 | 0.88 | 191 |
| dandelion | 0.93 | 0.90 | 0.92 | 263 |
| rose | 0.88 | 0.85 | 0.87 | 196 |
| sunflower | 0.89 | 0.85 | 0.87 | 184 |
| tulip | 0.83 | 0.91 | 0.87 | 246 |

> **Rose** is the hardest class for both frameworks (lowest recall) — visually closest to tulip; **dandelion** is the easiest.

---

## ⚙️ Installation
```bash
git clone https://github.com/krishnamaniyar2209/flower-image-classification.git
cd flower-image-classification
pip install -r requirements.txt
jupyter notebook Flower_Image_Classification.ipynb
```

---

## 🚀 Usage
1. Open the notebook in Jupyter or Google Colab (GPU recommended)
2. The dataset downloads automatically via `kagglehub`
3. Run all cells top to bottom — both models train, evaluate, and compare automatically

---

## 💡 Key Findings
- **Transfer learning converges fast** — frozen ResNet50 features reached ~86–88% test accuracy within a few epochs (TF early-stopped at epoch 5, best weights from epoch 2)
- **PyTorch (88.1%) slightly beat TensorFlow (86.0%)**, helped by flip + rotation augmentation
- **Rose** is the most-confused class (lowest recall in both frameworks); **dandelion** the easiest
- **EarlyStopping + ReduceLROnPlateau** prevented overfitting and stabilized convergence
- Frozen ImageNet features transfer well to a small (4.3K-image) flower dataset without fine-tuning the base

---

## 🛠️ Technologies Used
| Tool | Purpose |
|---|---|
| TensorFlow / Keras | Transfer learning (Step 3) |
| PyTorch / torchvision | Transfer learning (Step 4) |
| ResNet50 | Pre-trained base model |
| OpenCV | Image loading and resizing |
| scikit-learn | Metrics and data splitting |
| Matplotlib / Seaborn | Confusion-matrix visualization |

---

## 👤 Author

**Krishna Maniyar** — Data Scientist
- 🎓 Pace University — Seidenberg School of CSIS, MS in Data Science
- 📘 CS672: Introduction to Deep Learning (Fall 2025)
- 📧 krishnamaniyarkm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)

---

<p align="center">Made with ❤️ for CS672 @ Pace University</p>
