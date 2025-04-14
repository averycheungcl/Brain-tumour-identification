Here’s a comprehensive `README.md` for your project based on the code you provided:

---

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
pip install nibabel numpy torch matplotlib torchio tqdm scipy
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

1. **Set the dataset path** inside `main()`:
```python
base_path = os.path.expanduser("~/Downloads/Task01_BrainTumour (1)/Task01_BrainTumour/")
```

2. **Run the training**:
```bash
python Brain\ seg\ open\ all\ files.py
```

3. **Results**:
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

## 📈 Sample Visualization

MRI Slice | Ground Truth | Prediction
:--:|:--:|:--:
![MRI](results/segmentation_results_file_0.png)

---

## 📌 Notes

- The model processes 3D volumes in depth-wise chunks (default: 16 slices).
- You can change the number of files to process via `num_files_to_process` in `main()`.
- Make sure GPU is available for best performance. Otherwise, training will run on CPU.

---
