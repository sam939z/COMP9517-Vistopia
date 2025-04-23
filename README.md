
# SkyView Remote Sensing Image Classification

In this repository, we explore the SkyView aerial landscape dataset using five different image classification models, ranging from traditional machine learning methods to state-of-the-art deep learning architectures. Our goal is to provide an end-to-end, reproducible pipeline for remote sensing scene classification. This includes data preprocessing, feature extraction, model training, evaluation, and visualization.

All experiments are conducted on the [SkyView Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset), which contains 15 categories and 12,000 aerial images.

## 📦 Repository Structure

```bash
.
├── LBP-KNN.ipynb                  # LBP + KNN classification
├── SIFT-BoVW-SVM.ipynb          # SIFT + BoVW + SVM pipeline
├── ResNet18.ipynb                 # ResNet18 with transfer learning
├── EfficientNet-B0.ipynb         # EfficientNet-B0 and EfficientNet + SE
├── SE-ResNet50.ipynb   # SE-ResNet50 full pipeline
└── README.md
```

## 📌 Requirements

We recommend using Google Colab Pro or a local environment with GPU support.

Install the following Python packages:

```bash
pip install numpy opencv-python matplotlib scikit-learn tensorflow keras
```

> Optional (for traditional methods):
```bash
pip install scikit-image
```

| Framework        | Version |
|------------------|---------|
| Python           | 3.9+    |
| TensorFlow       | 2.10+   |
| Keras            | 2.10+   |
| scikit-learn     | 1.2+    |
| OpenCV           | 4.5+    |
| NumPy            | 1.22+   |

## 🧠 Model Descriptions & Performance

| Model              | Type             | Parameters | Accuracy | Strengths |
|--------------------|------------------|------------|----------|-----------|
| LBP + KNN          | Traditional ML   | Grid-based | 54.58%   | Fast, lightweight, but limited in fine-grained classes |
| SIFT + BoVW + SVM  | Traditional ML   | BoVW + C=10 + RBF | ~70% | Stronger than LBP; can handle sparse patterns |
| ResNet18           | Deep Learning    | Transfer Learning | ~96% | Good generalization with low complexity |
| EfficientNet-B0    | Deep Learning    | Compound Scaling | 97.75% | Balanced accuracy and speed |
| EfficientNet + SE  | Deep Learning    | + Attention | 97.42% | More stable on hard categories |
| SE-ResNet50        | Deep Learning    | + Attention | 98.00% | Best overall; strong feature learning capacity |

All models are trained using an 80/20 train-test split with balanced category distributions.

## 🏃‍♀️ How to Run

### 1. Run in Colab or Kaggle

You can run any of the following notebooks in Google Colab with minimal changes:

- `LBP-KNN.ipynb`  
- `SIFT-BoVW-SVM.ipynb`  
- `ResNet18.ipynb`  
- `EfficientNet-B0.ipynb`  
- `SE-ResNet50.ipynb`

Each notebook contains a complete pipeline including:
- Image loading
- Feature extraction or model initialization
- Training and validation
- Confusion matrix visualization
- Classification report generation

### 2. Run Locally

```bash
git clone https://github.com/yourusername/skyview-classification.git
cd skyview-classification
```

Install the required packages (see above), then launch the notebook of interest using Jupyter Lab or VSCode:

```bash
jupyter notebook ResNet18.ipynb
```

## 📊 Evaluation Metrics

All models are evaluated using:

- Accuracy
- Precision (Macro Average)
- Recall (Macro Average)
- F1-Score (Macro Average)
- Confusion Matrix (per class)

Ablation studies are also included for:
- LBP radius / K in KNN
- BoVW size / SVM kernel
- EfficientNet vs. EfficientNet + SE
- SE-ResNet variants (fine-tuning, data augmentation, etc.)

## 🧪 Notes

- All deep learning models use ImageNet pre-trained weights and are fine-tuned on the SkyView dataset.
- Data augmentation is applied using `ImageDataGenerator` with rotation, horizontal flip, zoom, and shear.
- For SE-based models, attention mechanisms are integrated at the channel level using the Squeeze-and-Excitation block.
- All results shown are reproducible using the default notebook settings.


