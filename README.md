# 🧠 Brain Tumor Segmentation using 3D U-Net

<div align="center">

**Robust 3D MRI brain tumor segmentation with enhanced U-Net architecture**

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![TorchIO](https://img.shields.io/badge/TorchIO-latest-brightgreen.svg)](https://torchio.readthedocs.io/)
[![NiBabel](https://img.shields.io/badge/NiBabel-latest-yellow.svg)](https://nipy.org/nibabel/)

</div>

## ✨ Features

- 🧠 **Enhanced 3D U-Net architecture** for volumetric segmentation
- 📊 **Multi-modal MRI support** with 4-channel input processing
- 🔄 **Advanced data augmentation** via TorchIO
- 📈 **Combined loss functions** (Dice + Focal + BCE) for optimal training
- 🧩 **Memory-efficient patch-wise training** for large 3D volumes
- 📊 **Comprehensive visualization tools** for segmentation results

## 📋 Requirements

- Python 3.x
- PyTorch
- TorchIO
- NiBabel
- NumPy
- Matplotlib
- SciPy
- Jupyter (optional)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# Install dependencies
pip install nibabel numpy torch matplotlib torchio tqdm scipy jupyter
```

### 2. Dataset Structure

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

### 3. Running the Application

#### Option 1: Python Script

```bash
# Set the dataset path inside main()
python "Brain seg open all files.py"
```

#### Option 2: Jupyter Notebook

```bash
jupyter notebook
# Open Brain_Tumor_Segmentation.ipynb
```

## 🔍 How It Works

1. **Data Preprocessing**
   - Load multi-modal MRI scans (T1, T2, FLAIR, etc.)
   - Normalize intensity values
   - Extract 3D patches for memory-efficient training

2. **Model Architecture**
   - 3D U-Net with customizable feature maps
   - Encoder path for feature extraction
   - Decoder path with skip connections for precise localization

3. **Training Process**
   - Combined loss functions for better boundary detection
   - Data augmentation for improved generalization
   - Learning rate scheduling and early stopping

4. **Segmentation & Post-processing**
   - Sliding window inference for full-volume prediction
   - Noise removal in predicted masks
   - Final segmentation in NIfTI format

5. **Evaluation & Visualization**
   - Dice coefficient and IoU metrics
   - Slice-by-slice visualization of results
   - Comparison plots of ground truth vs prediction

## 🧠 The Model

The project uses a customized 3D U-Net model designed for volumetric segmentation:

| Layer | Feature Maps | Resolution |
|-------|--------------|------------|
| Input | 4 (multi-modal) | Original |
| Encoder 1 | 16 | 1/2 |
| Encoder 2 | 32 | 1/4 |
| Encoder 3 | 64 | 1/8 |
| Bottleneck | 128 | 1/16 |
| Decoder 1 | 64 | 1/8 |
| Decoder 2 | 32 | 1/4 |
| Decoder 3 | 16 | 1/2 |
| Output | 1 (segmentation) | Original |

## 📁 Output

- Trained weights saved as `model_weights.pth`
- Visualizations stored in the `results/` directory
- Segmented NIfTI files for each input MRI

## 📊 Sample Visualization

MRI Slice | Ground Truth | Prediction
:--:|:--:|:--:

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
