# 🌸 Flower Image Classification with Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS672-blue)

> Fine-tuning a pre-trained **ResNet50** for 5-class flower classification, implemented twice: once in **TensorFlow/Keras** and once in **PyTorch**. Built for CS672: Introduction to Deep Learning at Pace University (Fall 2025).

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [What This Comparison Does and Does Not Show](#-what-this-comparison-does-and-does-not-show)
- [Limitations & Next Steps](#-limitations--next-steps)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🔬 Overview

This project implements **transfer learning** with a pre-trained ResNet50 (ImageNet weights, frozen backbone) to classify 5 flower species, built independently in **TensorFlow (Keras)** and **PyTorch**.

- ✅ Data preparation (load, resize, encode, stratified split)
- ✅ Pre-trained ResNet50 with frozen base and a custom classifier head
- ✅ TensorFlow training with EarlyStopping and ReduceLROnPlateau
- ✅ PyTorch training with data augmentation and manual early stopping
- ✅ Full evaluation: Accuracy, macro Precision, Recall, F1, per-class reports, confusion matrices
- ✅ Side-by-side comparison of the two implementations

---

## 📊 Dataset

**Source:** [Kaggle, Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

| Property | Details |
|---|---|
| Total Images | 4,317 |
| Classes | 5 (daisy, dandelion, rose, sunflower, tulip) |
| Per-class counts | dandelion 1,052 · tulip 984 · rose 784 · daisy 764 · sunflower 733 |
| Image Size | Loaded at 150×150 via OpenCV, resized to 224×224 for ResNet50 |

Class imbalance is mild, with the largest class (dandelion) at 1.43x the smallest (sunflower). No resampling or class weighting was applied.

### Data Split

| Set | Size | Share |
|---|---|---|
| Training | 2,589 | 60% |
| Validation | 648 | 15% |
| Test | 1,080 | 25% |

Produced by a stratified 75/25 train/test split, then a stratified 80/20 carve of validation out of training. Both splits use `random_state=42`. The identical split feeds both frameworks, so the test set is directly comparable.

---

## 📁 Project Structure
```
flower-image-classification/
│
├── Flower_Image_Classification.ipynb   # Main notebook
├── README.md                           # Documentation
├── requirements.txt                    # Dependencies
└── .gitignore                          # Excludes trained model weights

# Generated locally, not committed (see .gitignore):
#   flower_classifier_tensorflow.keras
#   flower_classifier_pytorch.pt
```

---

## 🔬 Methodology

### Step 1: Data Preparation
- Downloaded via `kagglehub`, images read with OpenCV and resized to 150×150
- Label encoding plus one-hot encoding (`to_categorical`)
- Stratified 75/25 train/test split, then stratified 80/20 validation carve from training

### Step 2: TensorFlow Implementation
```python
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_tensor=Input(shape=(224, 224, 3)))
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
output = Dense(5, activation='softmax')(x)
```
- Preprocessing: `tf.keras.applications.resnet50.preprocess_input`
- Optimizer: Adam (lr=0.001), Categorical Crossentropy, batch size 32
- Callbacks: `EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)` and `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)`
- **No augmentation**
- **Trainable head parameters: 525,829**

### Step 3: PyTorch Implementation
```python
torch_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
torch_model.fc = nn.Linear(torch_model.fc.in_features, 5)

for name, param in torch_model.named_parameters():
    if not name.startswith("fc."):
        param.requires_grad = False
```
- Preprocessing: ImageNet mean/std normalization
- Augmentation: `RandomHorizontalFlip()` and `RandomRotation(10)` on training data only
- Optimizer: Adam (lr=0.001), CrossEntropyLoss, batch size 32
- Manual early stopping on validation accuracy, patience 3
- **Trainable head parameters: 10,245**

### Training Behaviour

| | TensorFlow | PyTorch |
|---|---|---|
| Epoch budget | 20 | 20 |
| Epochs actually run | **5** | **13** |
| Best epoch | 2 (val acc 0.8997) | 10 (val acc 0.9105) |
| Stopping trigger | EarlyStopping, patience 3 | Manual early stopping, patience 3 |

TensorFlow peaked at epoch 2 and validation accuracy declined thereafter, while training accuracy kept climbing to 0.954. That divergence is overfitting on an unaugmented, 2,589-image training set. PyTorch, with augmentation, kept improving through epoch 10.

---

## 📈 Results

### Overall Comparison (macro-averaged)

| Implementation | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| TensorFlow | 0.860 | 0.862 | 0.854 | 0.857 |
| **PyTorch** | **0.881** | **0.883** | **0.878** | **0.880** |

All figures are macro averages on the shared 1,080-image test set.

### Per-Class Performance

**TensorFlow**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| daisy | 0.89 | 0.84 | 0.86 | 191 |
| dandelion | 0.89 | 0.91 | 0.90 | 263 |
| **rose** | 0.83 | **0.76** | **0.79** | 196 |
| sunflower | 0.89 | 0.85 | 0.87 | 184 |
| tulip | 0.81 | 0.91 | 0.86 | 246 |

**PyTorch**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| daisy | 0.89 | 0.87 | 0.88 | 191 |
| dandelion | 0.93 | 0.90 | 0.92 | 263 |
| rose | 0.88 | 0.85 | 0.87 | 196 |
| sunflower | 0.89 | 0.85 | 0.87 | 184 |
| tulip | 0.83 | 0.91 | 0.87 | 246 |

**The entire performance gap lives in one class.** Rose recall rises from 0.76 to 0.85 between the two runs. Every other class moves by 3 points or less, and dandelion and tulip are essentially unchanged. The 2.1-point difference in overall accuracy is almost entirely a rose story.

Rose is the weakest class in TensorFlow outright, and joint-weakest with sunflower in PyTorch. In both runs, **tulip has the lowest precision (0.81 and 0.83) alongside the highest recall (0.91)**, which is the signature of a class being over-predicted: the models reach for "tulip" when uncertain, and rose is the most likely donor given the visual similarity between the two.

---

## 🔍 What This Comparison Does and Does Not Show

The two implementations differ in **four** ways, not one. Only the frozen ResNet50 backbone and the data split are held constant.

| | TensorFlow | PyTorch |
|---|---|---|
| Augmentation | None | Flip + rotation |
| Classifier head | GAP → Dropout(0.3) → Dense(256) → Dense(5) | `Linear(2048, 5)` |
| Trainable parameters | 525,829 | 10,245 |
| Preprocessing | Caffe-style BGR mean subtraction | ImageNet mean/std |
| Epochs trained | 5 | 13 |

**This is therefore not a framework benchmark.** It compares two training recipes that happen to live in different libraries. Both run identical pretrained weights, so a properly controlled comparison would be expected to land within noise of itself. Any claim that "PyTorch is better than TensorFlow" is unsupported by this experiment.

**What it does show is more interesting.** The TensorFlow head carries **51 times more trainable parameters** and still finishes 2 points behind. On a frozen backbone with fewer than 3,000 training images, augmentation and training duration matter considerably more than classifier capacity. TensorFlow's larger head reached its best validation accuracy at epoch 2 and then overfit, while the augmented PyTorch run with a bare linear layer kept improving for ten.

---

## ⚠️ Limitations & Next Steps

1. **The framework comparison is confounded.** To make it a real benchmark, hold the head architecture, augmentation, preprocessing, and epoch budget identical and vary only the library.
2. **The backbone was never unfrozen.** Fine-tuning the final ResNet50 block at a low learning rate is the standard next step and would likely push accuracy past 92%.
3. **Single run, single seed.** All results come from one training run at `random_state=42`. Repeating across seeds would show whether the 2.1-point gap exceeds run-to-run variance. It may not.
4. **Confusion matrices are figures only.** The rose-to-tulip confusion is inferred from the precision and recall pattern rather than read directly. Printing the matrices would make the claim citable.
5. **Images are upscaled, not native.** Files are loaded at 150×150 and then resized up to 224×224, so the pipeline discards resolution before restoring it. Loading directly at 224×224 would preserve detail that may matter for rose and tulip discrimination.
6. **No class weighting.** The mild 1.43x imbalance is untreated. Unlikely to be the bottleneck, but untested.

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
1. Open the notebook in Jupyter or Google Colab (GPU strongly recommended)
2. The dataset downloads automatically via `kagglehub`
3. Run all cells top to bottom. Both models train, evaluate, and compare automatically
4. Trained weights are saved locally and excluded from version control by `.gitignore`

---

## 💡 Key Findings

- **Transfer learning converges fast.** Frozen ResNet50 features reached 86 to 88% test accuracy within a handful of epochs. TensorFlow's best weights came from epoch 2
- **Augmentation and training duration beat classifier capacity.** The TensorFlow head has 51x more trainable parameters (525,829 vs 10,245) and still finishes 2 points lower. Its unaugmented run peaked at epoch 2 while training accuracy climbed to 0.954, a clear overfitting signature
- **The gap is one class.** Rose recall improves from 0.76 to 0.85 between the two runs. Every other class moves by 3 points or less
- **Tulip is over-predicted in both runs**, holding the lowest precision and highest recall. Rose is the most plausible source of those false positives
- **The comparison is not a framework benchmark.** Four variables differ between the runs, so the result reflects training recipe rather than library
- **Frozen ImageNet features transfer well** to a small 4.3K-image flower dataset without unfreezing the base

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| TensorFlow / Keras | Transfer learning implementation |
| PyTorch / torchvision | Transfer learning implementation |
| ResNet50 (ImageNet) | Pre-trained frozen backbone |
| OpenCV | Image loading and resizing |
| scikit-learn | Stratified splitting and metrics |
| Matplotlib / Seaborn | Training curves and confusion matrices |
| kagglehub | Dataset retrieval |

---

## 👤 Author

**Krishna Maniyar**, Data Analyst
- 🎓 Pace University, Seidenberg School of CSIS, MS in Data Science
- 📘 CS672: Introduction to Deep Learning (Fall 2025)
- 📧 krishnamaniyarkm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)

---

<p align="center">Made with ❤️ for CS672 @ Pace University</p>
