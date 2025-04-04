# 🧠 Brain Tumor Detection using Deep Learning  

## 📌 Project Overview  
This project aims to detect brain tumors using a Convolutional Neural Network (CNN) model trained on MRI scan images. The model classifies images into tumor and non-tumor categories, improving diagnosis accuracy through deep learning techniques.

## Dataset
- The dataset consists of **253 MRI images** categorized into tumor and non-tumor classes.
- Images were resized to **128x128 pixels** for uniform processing.

## Data Preprocessing
### Resized Images
![image](https://github.com/user-attachments/assets/60f65c92-8549-4bfd-b231-fddcc95fa1e4)

### Normalization
- Images were normalized to scale pixel values between **0 and 1** to improve training efficiency.
- This prevents high variance issues and accelerates convergence.

**Before and After Normalization:**
![image](https://github.com/user-attachments/assets/ac3e9eee-2e96-4544-9d4e-13a1333db252)

### Data Augmentation
To increase dataset variability and reduce overfitting, the following transformations were applied:
- **Rotation**
- **Flipping**
- **Zooming**
- **Shearing**
- **Shifting**

**Augmented Images:**
![image](https://github.com/user-attachments/assets/1a92b6c8-482b-4788-a353-cdd801568fb3)


## Model Architecture
The CNN model consists of:
1. **Convolutional Layers** (Extracting spatial features)
2. **Max-Pooling Layers** (Downsampling)
3. **Batch Normalization** (Stabilizing training)
4. **Dropout** (Reducing overfitting)
5. **Fully Connected Layers** (Final classification)

**Model Architecture**
![image](https://github.com/user-attachments/assets/d5c07104-952e-47c7-9d49-38f31a5ab416)

## Model Training
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

 ![image](https://github.com/user-attachments/assets/930c9d73-9406-40b9-b09e-e10c43e4bfbc)


## Evaluation
- The model was evaluated using **confusion matrix, accuracy, precision, and recall**.
- Final accuracy achieved: **X%** (Replace with actual accuracy)

**Confusion Matrix:**
![image](https://github.com/user-attachments/assets/46bb2488-1a52-424b-925d-f9e2eecc9899)


## Results
- The model successfully distinguishes between tumor and non-tumor MRI images.
- Data augmentation significantly improved generalization.
  
![image](https://github.com/user-attachments/assets/ec1ad2e9-25f4-4243-ae3f-a05f7f490547)

## 🎯 Model Performance  

### **Before Hyperparameter Tuning:**  
- **Validation Loss:** 0.5617  
- **Validation Accuracy:** 72.55%  

### **After Hyperparameter Tuning:**  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step  
- **Training Accuracy:** 76.00%  
- **Validation Accuracy:** 75.99%  

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step  
- **Test Accuracy:** 88.46%

## Future Improvements
- Implementing **transfer learning** with pre-trained models (VGG16, ResNet, etc.).
- Improving dataset quality and balance.
- Deploying the model as a web application.

## References
- Dataset Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Deep Learning Framework: TensorFlow/Keras

---
🚀 **Author:** Melody Bonareri  
📌 **GitHub Repository:** https://github.com/bonareri/Brain-Tumor-Detection/edit/main/README.md
