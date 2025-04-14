# 🧠 Brain Tumor Segmentation using 3D U-Net

This project implements a robust pipeline for 3D MRI brain tumor segmentation using an enhanced 3D U-Net architecture. The code handles data preprocessing, model training, evaluation, and visualization with support for multiple files.

## 📁 Project Structure

- **Data Input**: Brain MRI scans and their corresponding masks in NIfTI format (`.nii.gz`).
- **Model**: A customized 3D U-Net.
- **Training**: Dice + Focal + BCE loss, with data augmentation and learning rate scheduling.
- **Output**: Trained model weights, segmentation visualizations, and saved prediction volumes.

---

## 🚀 Features

- ✅ 3D U-Net with customizable feature maps  
- ✅ Augmentation using `TorchIO`  
- ✅ Patch-wise training for memory efficiency  
- ✅ Dice, Focal, and BCE loss for better segmentation performance  
- ✅ Logging and visualization support  
- ✅ Post-processing for noise removal in predicted masks

---

## 🧩 Requirements

Make sure you have the following libraries installed:

```bash
pip install nibabel numpy torch matplotlib torchio tqdm scipy jupyter
```

---

## 📂 Dataset Structure

Expected directory structure for input data:

```
Task01_BrainTumour/
├── imagesTr/
│   ├── BRATS_001.nii.gz
│   └── ...
├── labelsTr/
│   ├── BRATS_001.nii.gz
│   └── ...
```

---

## 🛠️ How to Run

### Option 1: Python Script

1. **Set the dataset path** inside `main()`:
```python
base_path = os.path.expanduser("~/Downloads/Task01_BrainTumour (1)/Task01_BrainTumour/")
```

2. **Run the training**:
```bash
python Brain\ seg\ open\ all\ files.py
```

### Option 2: Jupyter Notebook

1. **Launch Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Open the notebook**:
```
Brain_Tumor_Segmentation.ipynb
```

3. **Set the dataset path** in the notebook configuration cell:
```python
base_path = os.path.expanduser("~/Downloads/Task01_BrainTumour (1)/Task01_BrainTumour/")
```

4. **Run all cells** or execute them step by step to train the model and visualize results.

### Results

- Trained weights are saved as `model_weights.pth`
- Visualizations saved under `results/`
- Segmented NIfTI files for each input MRI

---

## 🔍 Key Components

### 🧠 `UNet3D`
- Standard encoder-decoder U-Net with skip connections and dropout
- Designed for multi-modal MRI (4 channels)

### 🧪 `MRIDataset`
- Dynamically extracts volumetric patches from MRI scans
- Applies augmentation during training

### 📉 `Loss Functions`
- Combines BCE, Dice, and Focal loss for robust training

### 🎯 `Training Loop`
- Tracks training/validation metrics (loss, Dice, IoU)
- Early stopping and learning rate reduction on plateau

### 👁️ `visualize_results`
- Saves comparative plots of MRI slices, ground truth, and predictions

---

## 📊 Notebook Features

The Jupyter notebook provides additional benefits:

- **Interactive Experimentation**: Easily modify parameters and see results
- **Visualizations**: Real-time plots of training progress
- **Step-by-Step Execution**: Run each component separately for easier debugging
- **Markdown Documentation**: Detailed explanations of each code section
- **Interactive Parameter Tuning**: Use widgets to adjust hyperparameters

---

## 📖 Technical Documentation

For developers interested in understanding the implementation details, architecture decisions, and technical insights behind this project, see the [TECHNICAL_DETAILS.md](./TECHNICAL_DETAILS.md) file. This document provides in-depth explanations of:

- Data preprocessing techniques
- 3D U-Net architecture specifics
- Loss function design and rationale
- Training optimizations
- Inference strategy with sliding window approach
- Visualization methods
- Potential extensions for production use

This technical documentation can serve as a reference for developers or as preparation material for technical discussions and interviews.

---

## 📈 Sample Visualization

MRI Slice | Ground Truth | Prediction
:--:|:--:|:--:
![MRI](results/segmentation_results_file_0.png)

---

## 📌 Notes

- The model processes 3D volumes in depth-wise chunks (default: 16 slices).
- You can change the number of files to process via `num_files_to_process` in the configuration.
- Make sure GPU is available for best performance. Otherwise, training will run on CPU.
- The Jupyter notebook includes more detailed documentation and visualizations than the script version.

---
