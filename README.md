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
- [Confusion Matrices](#-confusion-matrices)
- [What This Comparison Does and Does Not Show](#-what-this-comparison-does-and-does-not-show)
- [Known Issue: Channel Order](#-known-issue-channel-order)
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
- **Trainable head parameters: 525,829** (2048×256+256, plus 256×5+5)

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
- **Trainable head parameters: 10,245** (2048×5+5)

### Training Behaviour

| | TensorFlow | PyTorch |
|---|---|---|
| Epoch budget | 20 | 20 |
| Epochs actually run | **5** | **13** |
| Best epoch | 2 (val acc 0.8997) | 10 (val acc 0.9105) |
| Stopping trigger | EarlyStopping, patience 3 | Manual early stopping, patience 3 |

TensorFlow peaked at epoch 2 and never exceeded that validation accuracy again (0.8827, 0.8642, then a partial recovery to 0.8920), while training accuracy kept climbing to 0.954. That divergence is overfitting on an unaugmented, 2,589-image training set. PyTorch, with augmentation, kept improving through epoch 10.

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

---

## 🔢 Confusion Matrices

Rows are actual, columns predicted.

**TensorFlow** — 929 / 1,080 correct

| actual ↓ / pred → | daisy | dandelion | rose | sunflower | tulip |
|---|---|---|---|---|---|
| **daisy** | **160** | 18 | 2 | 4 | 7 |
| **dandelion** | 5 | **239** | 5 | 10 | 4 |
| **rose** | 13 | 1 | **148** | 2 | **32** |
| **sunflower** | 2 | 7 | 9 | **157** | 9 |
| **tulip** | 0 | 3 | 14 | 4 | **225** |

**PyTorch** — 952 / 1,080 correct

| actual ↓ / pred → | daisy | dandelion | rose | sunflower | tulip |
|---|---|---|---|---|---|
| **daisy** | **166** | 11 | 0 | 9 | 5 |
| **dandelion** | 9 | **238** | 2 | 5 | 9 |
| **rose** | 5 | 2 | **167** | 1 | **21** |
| **sunflower** | 6 | 4 | 5 | **157** | 12 |
| **tulip** | 1 | 1 | 16 | 4 | **224** |

### The rose–tulip confusion, quantified

| | TensorFlow | PyTorch |
|---|---|---|
| Rose misclassified as tulip | **32** | **21** |
| Rose's other errors combined | 16 | 8 |
| Share of rose errors going to tulip | **67%** | **72%** |
| Tulip's total false positives | 52 | 47 |
| Share of those originating from rose | **62%** | **45%** |

Tulip carries the lowest precision (0.81 and 0.83) alongside the highest recall (0.91) in both runs — the signature of a class being over-predicted. The matrices confirm rose is the dominant donor of those false positives in both frameworks.

### The gap is almost entirely one class

Correct predictions per class, TensorFlow → PyTorch:

| Class | TF | PyTorch | Δ |
|---|---|---|---|
| **rose** | 148 | 167 | **+19** |
| daisy | 160 | 166 | +6 |
| sunflower | 157 | 157 | 0 |
| dandelion | 239 | 238 | −1 |
| tulip | 225 | 224 | −1 |
| **Total** | **929** | **952** | **+23** |

**19 of the 23 additional correct predictions are roses — 83% of the entire improvement.** The 2.1-point accuracy difference is a rose story, not a general one.

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

> One caveat on the preprocessing row: both pipelines share the channel-order defect described below, so preprocessing differentiates the two runs less than the table suggests.

---

## 🐛 Known Issue: Channel Order

**Both pipelines feed BGR images to models that expect RGB.**

`cv2.imread()` returns BGR. The notebook's display cells correctly call `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`, but that conversion never reaches the training path:

- **TensorFlow** — `resnet50.preprocess_input` uses `mode='caffe'`, which expects RGB input, reverses it to BGR, and subtracts BGR-ordered ImageNet means. Given BGR input it emits RGB and misaligns the mean subtraction.
- **PyTorch** — `transforms.ToPILImage()` interprets a 3-channel array as RGB. With BGR data, red and blue swap before ImageNet RGB normalization, and torchvision's ResNet50 expects true RGB.

The one-line fix, at load time:

```python
img_array = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
```

**Why it matters:**

1. **86–88% is a floor, not a ceiling.** ImageNet-pretrained features degrade measurably on channel-swapped input.
2. **It affects both runs identically**, so the TensorFlow-vs-PyTorch comparison above remains internally consistent — but it removes preprocessing as a genuine differentiator between them.
3. **It is a candidate explanation for the rose–tulip confusion specifically.** Rose and tulip are separated largely by colour, and red↔blue is exactly the channel pair being swapped. Rerunning with correct channel order and re-reading the rose row of the confusion matrix is a direct test of that hypothesis, and the most interesting open question in the project.

---

## ⚠️ Limitations & Next Steps

1. **Channel order is wrong in both pipelines.** See above. Fixing it and rerunning is the highest-value next step, ahead of any architectural change.
2. **The framework comparison is confounded.** To make it a real benchmark, hold the head architecture, augmentation, preprocessing, and epoch budget identical and vary only the library.
3. **The backbone was never unfrozen.** Fine-tuning the final ResNet50 block at a low learning rate is the standard next step and would likely push accuracy past 92%.
4. **Single run, single seed.** All results come from one training run at `random_state=42`. Repeating across seeds would show whether the 2.1-point gap exceeds run-to-run variance. It may not.
5. **Confusion matrices are plotted but not printed.** The counts in this README were read off the figures. Adding `print(confusion_matrix(...))` alongside the heatmaps would make them machine-readable and reproducible.
6. **Images are upscaled, not native.** Files are loaded at 150×150 and then resized up to 224×224, so the pipeline discards resolution before restoring it. Loading directly at 224×224 would preserve detail that may matter for rose and tulip discrimination.
7. **No class weighting.** The mild 1.43x imbalance is untreated. Unlikely to be the bottleneck, but untested.

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
- **The gap is one class.** Rose accounts for **19 of the 23** additional correct predictions PyTorch makes — 83% of the total improvement. Every other class moves by one to six images
- **Rose is misread as tulip.** 32 of TensorFlow's 48 rose errors land on tulip (67%), and 21 of PyTorch's 29 (72%). Tulip holds the lowest precision and highest recall in both runs, the signature of over-prediction
- **Both pipelines feed BGR to RGB-expecting models**, so the reported accuracies are a floor. Colour-based class pairs like rose and tulip are the most likely victims
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
- 📧 maniyarkrishnakm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)

---

<p align="center">Made with ❤️ for CS672 @ Pace University</p>
