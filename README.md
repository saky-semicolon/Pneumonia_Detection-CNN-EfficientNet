# Pneumonia Diagnosis Using Deep Learning: A CNN and Transfer Learning Approach

## Project Description  
This project automates pneumonia detection from chest X-ray images using deep learning. It compares a **custom Convolutional Neural Network (CNN)** with transfer learning models **EfficientNetB0** and **EfficientNetB3** to identify the most effective approach for accurate and rapid diagnosis. The goal is to reduce reliance on manual X-ray interpretation, especially in resource-limited healthcare settings.

**Key Features**:  
- Dataset: 5,856 chest X-ray images (Normal vs. Pneumonia).  
- Preprocessing: Augmentation, normalization, and rescaling.  
- Evaluation Metrics: Accuracy, precision, recall, F1-score, and ROC-AUC.  

---

## Dataset  
The dataset is sourced from [Kaggle's Chest X-ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).  

**Dataset Composition**:  
| Split         | Images | Percentage |  
|---------------|--------|------------|  
| **Training**  | 5,216  | 80%        |  
| **Test**      | 624    | 20%        |  
| **Validation**| 16     | 0.27%      |  

**Classes**:  
- `Normal`: X-rays of healthy lungs.  
- `Pneumonia`: X-rays infected with bacterial or viral pneumonia.  

**Preprocessing Steps**:  
1. **Data Augmentation**: Random rotations, flips, brightness adjustments.  
2. **Normalization**: Pixel values scaled to [0, 1].  
3. **Resizing**: Images resized to 224x224 pixels.  

---

## Models  
### 1. Custom Convolutional Neural Network (CNN)  
**Architecture**:  
| Layer Type     | Parameters                          |  
|----------------|-------------------------------------|  
| Conv2D         | 64, 128, 256, 512 filters          |  
| MaxPooling     | Pool size (2,2)                    |  
| Dropout        | Rate: 0.5 (50% neurons disabled)   |  
| Dense          | 256, 128, 64 units + ReLU activation |  
| Output Layer   | 1 unit with Sigmoid activation     |  

**Training**:  
- **Optimizer**: Adam  
- **Loss Function**: Binary Cross-Entropy  
- **Training Time**: ~5 minutes  

### 2. Transfer Learning Models  
**EfficientNetB0** and **EfficientNetB3**:  
- **Pretrained**: On ImageNet (1,000 classes).  
- **Fine-Tuning**: Global Average Pooling + Dense layers added.  
- **Training**:  
  - **Optimizer**: Adam  
  - **Loss Function**: Binary Cross-Entropy  
  - **Training Time**: ~6 minutes (B0), ~8 minutes (B3)  

---

## Installation  
### Dependencies  
- Python 3.7+  
- TensorFlow 2.x  
- Keras  
- OpenCV  
- Matplotlib  
- NumPy  
- Pandas  

Install via:  
```bash
pip install tensorflow keras opencv-python matplotlib numpy pandas
```

## Key Findings

- CNN outperformed transfer learning models in accuracy, speed, and overall metrics.
- EfficientNet models showed lower accuracy but potential for complex feature extraction.
- Training time increased with model complexity (B3 > B0 > CNN).

## Conclusion
The custom CNN is recommended for deployment in resource-limited environments due to its 74% accuracy and rapid training (~5 minutes). While EfficientNet models underperformed here, further fine-tuning with larger datasets could enhance their utility. This project highlights the potential of deep learning to automate pneumonia diagnosis and reduce healthcare disparities.

## Author: S M Asiful Islam Saky
Thanks!
