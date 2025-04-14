# Technical Documentation: Brain Tumor Segmentation with 3D U-Net

This document provides detailed technical explanations of the implementation, architectural choices, and optimization strategies used in this brain tumor segmentation project.

## Table of Contents
1. [Data Preprocessing and Loading](#1-data-preprocessing-and-loading)
2. [3D U-Net Architecture](#2-3d-u-net-architecture)
3. [Loss Functions and Metrics](#3-loss-functions-and-metrics)
4. [Training Loop with Advanced Features](#4-training-loop-with-advanced-features)
5. [Inference Strategy](#5-inference-strategy)
6. [Visualization for Quality Assessment](#6-visualization-for-quality-assessment)
7. [Main Function Flow](#7-main-function-flow)
8. [Key Technical Insights](#8-key-technical-insights)

---

## 1. Data Preprocessing and Loading

### Function: `load_and_preprocess_mri(file_path, is_mask=False, target_shape=None)`

This function loads and preprocesses MRI data and masks from .nii files (NIfTI format) to prepare them for training.

#### Handling NaN Values
- **Implementation**: Replace NaN values with 0 using `np.nan_to_num(mri_data)`
- **Rationale**: NaN values can arise due to missing or corrupted data in medical imaging datasets. Replacing them with 0 ensures numerical stability during tensor operations and avoids errors during normalization or model training.

#### Normalization (for MRI Data Only)
- **Implementation**: For MRI data (`is_mask=False`), normalize intensity values to have zero mean and unit variance: `(mri_data - mean) / (std + 1e-7)`
- **Rationale**: MRI intensity values can vary widely across different scans due to scanner differences or acquisition protocols. Normalization ensures the data is on a consistent scale, which helps the model converge faster and improves performance.
- **Note**: Masks are binary (0 or 1) and represent segmentation labels, not intensity values. Normalizing them would distort their discrete nature, so this step is skipped for masks.

#### Tensor Shape Transformations

##### For Masks (`is_mask=True`):
- **Add Channel Dimension**: If the mask is 3D ([depth, height, width]), add a channel dimension using `unsqueeze(0)` to make it [1, depth, height, width].
- **Binarization**: Binarize the mask by thresholding at 0.5 (`mri_tensor = (mri_tensor > 0.5).float()`), converting intermediate values to 0 or 1.

##### For MRI Data (`is_mask=False`):
- **Handle 4D Data**: If the MRI data is 4D ([depth, height, width, channels]), permute the dimensions to [channels, depth, height, width] using `permute(3, 0, 1, 2)`.
- **Add Channel Dimension**: If the MRI data is 3D ([depth, height, width]), add a channel dimension using `unsqueeze(0)`.
- **Add Batch Dimension**: For both MRI and mask data, add a batch dimension using `unsqueeze(0)`, resulting in [1, channels, depth, height, width].

#### Resize Operation
- **Implementation**: Check if tensor's spatial dimensions match the `target_shape`. If not, resize using `F.interpolate`:
  - For masks: `mode="nearest"` to preserve discrete binary labels
  - For MRI data: `mode="trilinear"` for smooth interpolation
- **Rationale**: Resizing ensures all inputs to the model have consistent dimensions, critical for batch processing and model training. The interpolation mode choice is based on the nature of the data: discrete for masks, continuous for MRI intensities.

#### Logging
- **Implementation**: Log slices with non-zero data to ensure the MRI data contains meaningful information and to debug potential issues with empty slices.

---

## 2. 3D U-Net Architecture

### Class: `UNet3D(nn.Module)`

The UNet3D class implements a 3D U-Net architecture, a popular encoder-decoder model for 3D medical image segmentation.

#### Components of a Convolutional Block
- **Convolution (`Conv3d`)**: Each block contains two 3D convolutional layers with kernel size 3 and padding 1. Convolutions extract spatial features (edges, textures, shapes) by applying filters to the input volume.
- **BatchNorm3d**: Normalizes the output of each convolution to stabilize training and reduce internal covariate shift.
- **ReLU Activation**: Introduces non-linearity by setting negative values to 0, helping the model learn complex patterns, avoiding saturation issues, and introducing sparsity.
- **Dropout**: Applied with a rate of 0.3 to prevent overfitting by randomly setting a fraction of activations to 0 during training.

#### Encoder-Decoder Structure with Skip Connections

##### Encoder (Downsampling Path)
- **Purpose**: Extracts features by reducing spatial dimensions while increasing feature channels
- **Implementation**: Each level consists of a convolutional block followed by `MaxPool3d` (kernel size 2, stride 2)
- **Feature Progression**: Channels increase progressively ([16, 32, 64, 128]), capturing higher-level features at the cost of spatial detail

##### Decoder (Upsampling Path)
- **Purpose**: Reconstructs spatial dimensions by upsampling feature maps
- **Implementation**: Each level uses `ConvTranspose3d` (kernel size 2, stride 2) to double spatial dimensions, followed by a convolutional block
- **Feature Progression**: Channels decrease symmetrically ([128, 64, 32, 16]), restoring original resolution while combining features

##### Bottleneck
- **Purpose**: Deepest layer processing the most abstract features
- **Implementation**: Convolutional block with largest number of feature channels (256)
- **Importance**: Preserves spatial information at low resolution to enable accurate reconstruction

##### Skip Connections
- **Implementation**: Connect encoder and decoder by concatenating feature maps
- **Importance**: Allow decoder to combine high-level context with fine-grained spatial details, improving segmentation accuracy for precise tumor boundary delineation
- **Technical Detail**: Concatenate encoder feature maps with upsampled decoder feature maps along the channel dimension

#### Feature Dimensions Progression
- **Encoder**: Spatial dimensions halved at each level (e.g., [16, 128, 128, 128] → [32, 64, 64, 64] → [64, 32, 32, 32]), feature channels doubled ([16, 32, 64, 128])
- **Bottleneck**: Deepest level, dimensions approximately [256, 8, 8, 8]
- **Decoder**: Spatial dimensions doubled at each level, feature channels halved
- **Final Output**: Final convolution maps feature maps to desired output channels (1 for binary segmentation)

#### Handling Size Mismatches in Skip Connections
- **Issue**: During upsampling, decoder feature maps may not match spatial dimensions of corresponding encoder feature maps due to rounding errors
- **Solution**: Use `F.interpolate` to resize decoder feature maps:
  ```python
  if x.shape != skip_connection.shape:
      x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=False)
  ```
- **Rationale**: Trilinear interpolation ensures smooth resizing of 3D volumes, preserving feature continuity

---

## 3. Loss Functions and Metrics

### Classes/Functions: `DiceLoss(nn.Module)`, `FocalLoss(nn.Module)`, `calculate_metrics(predictions, targets, threshold=0.5)`

#### Dice Coefficient for Segmentation
- **Definition**: Measures overlap between predicted segmentation and ground truth: `2 * |A ∩ B| / (|A| + |B|)`
- **Implementation**: In `DiceLoss`, compute Dice score and return `1 - Dice` as the loss to minimize
- **Importance**: Robust to class imbalance (common in medical imaging where tumor regions are small compared to background)

#### Combined Loss Function
- **Implementation**:
  ```python
  return 0.2 * bce + 0.5 * dice + 0.3 * focal
  ```
- **Components**:
  - **BCE Loss**: Measures pixel-wise error, ensuring accurate probability predictions for each voxel
  - **Dice Loss**: Focuses on overlap, directly optimizing the segmentation metric
  - **Focal Loss**: Addresses class imbalance by down-weighting easy examples (parameters: `alpha=0.25`, `gamma=2.0`)
- **Weight Rationale**: Dice Loss weighted highest (0.5) because it directly optimizes the primary evaluation metric

#### Metrics for Medical Imaging
- **Dice Score**: Measures overlap as discussed; higher values indicate better segmentation
- **IoU (Intersection over Union)**: Defined as `|A ∩ B| / |A ∪ B|`; stricter than Dice, penalizing both false positives and negatives
- **Implementation**: In `calculate_metrics`, compute Dice and IoU by binarizing predictions (`> threshold`) and comparing to ground truth
- **Importance**: In medical imaging, these metrics directly measure segmentation quality, impacting clinical decisions (e.g., identifying tumor boundaries for surgery)

---

## 4. Training Loop with Advanced Features

### Function: `training_loop(model, params)`

#### Mixed Precision Training
- **Implementation**: Use PyTorch's automatic mixed precision (AMP) with `torch.amp.autocast` and `GradScaler` when training on GPU
- **Benefit**: Reduces memory usage and speeds up training on GPUs by using both 16-bit and 32-bit floating-point arithmetic
- **Importance**: Particularly valuable for 3D medical imaging with large data volumes and memory constraints
- **Technical Detail**: Wrap forward pass in `torch.amp.autocast`, scale loss with `scaler.scale`, update optimizer with `scaler.step` and `scaler.update`

#### Early Stopping
- **Implementation**: Stop training if validation Dice score does not improve for 25 consecutive epochs
- **Benefit**: Prevents overfitting to training data and saves computational resources
- **Technical Detail**: Track best validation score and epochs without improvement

#### Learning Rate Scheduling
- **Implementation**: Use `ReduceLROnPlateau` to adjust learning rate based on validation loss
- **Parameters**: If validation loss doesn't decrease for 10 epochs (`patience=10`), reduce learning rate by factor of 0.5 (`factor=0.5`), down to minimum of 1e-6 (`min_lr=1e-6`)
- **Benefit**: Helps model escape local minima and converge to better solution by reducing learning rate when loss plateaus

#### Tracking and Saving the Best Model
- **Implementation**: Track best validation Dice score and save model when new best score is achieved using `torch.save`
- **Benefit**: Ensures most effective version is available for inference, especially since validation performance may later degrade due to overfitting

---

## 5. Inference Strategy

### Function: `segment_mri(mri_tensor, model, device=None, threshold=0.5)`

#### Sliding Window Approach with Overlap
- **Implementation**: Process MRI volume in smaller chunks (`volume_depth=16`) using sliding window across depth dimension with overlap (stride of `volume_depth // 2 = 8`)
- **Rationale**: Entire MRI volume (e.g., [1, 4, 155, 240, 240]) may be too large for GPU memory; sliding window breaks it into manageable chunks while ensuring full coverage

#### Handling Overlapping Predictions
- **Implementation**: Average predictions in overlapping regions using `output_tensor` and `count_tensor`
  - For each chunk, add model's predictions to `output_tensor` and increment `count_tensor` for corresponding voxels
  - Final prediction is `output_tensor / count_tensor`
- **Benefit**: Averaging reduces boundary artifacts and improves consistency across volume, especially at chunk boundaries

#### Threshold for Binary Segmentation
- **Implementation**: Apply threshold of 0.5 to sigmoid-activated predictions for binary segmentation
- **Rationale**: Model outputs logits converted to probabilities using `torch.sigmoid`; threshold of 0.5 is standard for binary classification
- **Flexibility**: Threshold is parameterized, adjustable based on validation performance (e.g., 0.7 to reduce false positives)

#### Post-Processing
- **Implementation**: Apply `post_process_mask` to remove small regions (< 50 voxels) using connected component analysis
- **Benefit**: Removes noise and small false positives, improving segmentation quality

---

## 6. Visualization for Quality Assessment

### Function: `visualize_results(mri_data, true_mask, predicted_mask, slice_indices=None, save_path=None)`

#### Visualization Approach
- **Implementation**: Display original MRI, ground truth mask, and predicted mask side by side in 3-row subplot
- **Benefit**: Side-by-side visualization enables qualitative assessment of model performance, making it easy to spot discrepancies

#### Slice Selection Strategy
- **Implementation**: Select slices around middle of volume (`middle_idx = depth // 2`) with offsets
- **Filtering**: Ensure selected slices contain meaningful data by checking for non-zero voxels (≥ 100 non-zero pixels)
- **Rationale**: Middle slices typically contain most brain tissue in MRI volumes; filtering ensures meaningful visualization

#### Handling Different Data Dimensions
- **Implementation**: Handle varying input dimensions by squeezing or selecting channels as needed
- **Rationale**: MRI data may have multiple channels (T1, T2, FLAIR), but for visualization, focus on one channel for simplicity

#### Saving and Display
- **Implementation**: Save visualization to file (`save_path`) and display using `plt.show`
- **Benefit**: Saving allows tracking progress over epochs; displaying provides immediate feedback during development

---

## 7. Main Function Flow

### Function: `main()`

#### Overall Pipeline
1. **Data Loading**: Process .zip file containing .nii.gz files, extract and decompress files, load each image-label pair
2. **Model Creation**: Initialize UNet3D model with specified feature dimensions ([16, 32, 64, 128])
3. **Training**: Train model with mixed precision, early stopping, and learning rate scheduling
4. **Inference**: Load best model and segment entire MRI volume
5. **Visualization and Saving**: Visualize results and save segmented data

#### Data Preparation for 3D Volumes
- **Implementation**: Use `MRIDataset` to split 3D volume into smaller chunks (`volume_depth=16`) with overlap (`overlap=4`) for training
- **Benefit**: Reduces memory usage while ensuring continuity through overlapping sections, helping model learn spatial relationships

#### Model Configuration and Hyperparameters
- **Model Features**: [16, 32, 64, 128], balancing model capacity and computational efficiency
- **Optimizer**: Adam with learning rate 1e-3 and weight decay 1e-5 to prevent overfitting
- **Loss Weights**: [0.2, 0.5, 0.3] for BCE, Dice, and Focal Loss, prioritizing Dice for segmentation performance
- **Training**: Up to 200 epochs with early stopping (patience=25) to prevent overfitting

---

## 8. Key Technical Insights

### Why 3D U-Net?
- **Architecture Rationale**: 3D U-Net is specifically designed for volumetric data like MRI scans, capturing 3D spatial relationships
- **Advantage Over 2D**: Leverages full 3D context, avoiding need to process slices independently
- **Comparison**: Compared to architectures like DeepLab, U-Net's simplicity and effectiveness for medical imaging make it a standard choice

### Loss Function Combination Rationale
- **Multi-Objective Approach**: Combining BCE, Dice, and Focal Loss addresses different aspects of segmentation:
  - Pixel-wise accuracy (BCE)
  - Overlap/region-based accuracy (Dice)
  - Class imbalance handling (Focal)
- **Weight Distribution**: Prioritize Dice (0.5) as primary evaluation metric, with supporting weights for BCE (0.2) and Focal (0.3)

### Challenges of 3D Medical Imaging
- **Memory Constraints**: 3D volumes are large, requiring sliding window approach with overlap for inference and chunking during training
- **Data Normalization**: Normalize MRI data to handle intensity variations, ensuring model generalization across different scans
- **Class Imbalance**: Address with specialized loss functions (Dice and Focal)

### Sliding Window with Overlap Benefits
- **Quality Improvement**: Overlapping windows ensure continuity at chunk boundaries, reducing artifacts
- **Smoothing Effect**: Averaging predictions in overlapping regions improves overall segmentation quality 
- **Trade-Offs**: Increases computational cost due to redundant processing but necessary for high-quality results in large volumes

### Potential Extensions for Production
- **Multi-Class Segmentation**: Extend model for different tumor regions by increasing output channels and using softmax-based loss
- **Batch Processing**: Use data loaders to process multiple samples in batches, pre-train on larger dataset, or apply transfer learning
- **Memory Optimization**: Use gradient checkpointing or reduce model feature dimensions for efficiency
