#yellow is tumour region

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
import logging
import torchio as tio
from scipy.ndimage import label

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standardize randomization
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Step 1: Load and Preprocess MRI Data
def load_and_preprocess_mri(file_path, is_mask=False, target_shape=None):
    try:
        mri_scan = nib.load(file_path)
        mri_data = mri_scan.get_fdata()
    except Exception as e:
        logger.error(f"Error loading MRI file {file_path}: {e}")
        raise
    mri_data = np.nan_to_num(mri_data) # take care of NaN values
    if not is_mask: 
        non_zero_slices = np.where(np.any(mri_data > 0, axis=(0, 1, 2)))[0]
        if len(non_zero_slices) > 0:
            logger.info(f"Slices with non-zero data: {non_zero_slices[0]} to {non_zero_slices[-1]}")
        else:
            logger.warning("No slices with non-zero data found!")
        mean, std = np.mean(mri_data), np.std(mri_data)
        if std > 0:
            mri_data = (mri_data - mean) / (std + 1e-7)
    mri_tensor = torch.from_numpy(mri_data).float()
    if is_mask:
        if mri_tensor.dim() == 3:
            mri_tensor = mri_tensor.unsqueeze(0)
        unique_vals = np.unique(mri_data)
        logger.info(f"Mask unique values: {unique_vals}")
        mri_tensor = (mri_tensor > 0.5).float()
    else:
        if mri_tensor.dim() == 4:
            mri_tensor = mri_tensor.permute(3, 0, 1, 2)
        elif mri_tensor.dim() == 3:
            mri_tensor = mri_tensor.unsqueeze(0)
    if mri_tensor.dim() == 4:
        mri_tensor = mri_tensor.unsqueeze(0)
    if target_shape is not None and mri_tensor.shape[2:] != target_shape:
        logger.info(f"Resizing from {mri_tensor.shape[2:]} to {target_shape}")
        mode = "nearest" if is_mask else "trilinear"
        mri_tensor = F.interpolate(mri_tensor, size=target_shape, mode=mode,
                                   align_corners=False if mode == "trilinear" else None)
    logger.info(f"Loaded {'mask' if is_mask else 'MRI'} with shape {mri_tensor.shape}")

    '''we have two seperate statements for checking whether the file is a mask or image since they have different dimensions
    with mask having dimension [1,1,240,240,155] but image having [1,4,240,240,155] 
    
    
    '''
    return mri_tensor, mri_scan

# Step 2: Enhanced 3D U-Net Model
class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features=[16, 32, 64, 128]):
        super(UNet3D, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        for feature in features:
            self.downs.append(self._make_conv_block(in_channels, feature, dropout=0.3))
            in_channels = feature
        
        self.bottleneck = self._make_conv_block(features[-1], features[-1]*2, dropout=0.3)
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self._make_conv_block(feature*2, feature, dropout=0.3))
        
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=False)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)
    
    def _make_conv_block(self, in_channels, out_channels, dropout=0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

# Step 3: Dataset with Augmentation
class MRIDataset(Dataset):
    def __init__(self, mri_tensor, mask_tensor, volume_depth=16, overlap=0, augment=False):
        self.data = mri_tensor  # [N, 4, D, H, W]
        self.masks = mask_tensor  # [N, 1, D, H, W]
        self.num_files = self.data.shape[0]
        self.depth = self.data.shape[2]
        self.volume_depth = volume_depth
        self.overlap = overlap
        self.stride = volume_depth - overlap
        self.augment = augment and torch.cuda.is_available()
        self.transforms = tio.Compose([
            tio.RandomFlip(axes=('LR',), p=0.5),
            tio.RandomAffine(degrees=10, translation=5, p=0.5),
            tio.RandomElasticDeformation(num_control_points=7, max_displacement=2.0, p=0.3),
            tio.RandomNoise(std=0.01, p=0.3),
        ]) if self.augment else None
        
        self.start_positions = list(range(0, self.depth - self.volume_depth + 1, self.stride))
        if self.start_positions[-1] + self.volume_depth < self.depth:
            self.start_positions.append(self.depth - self.volume_depth)
        self.total_items = self.num_files * len(self.start_positions)

    def __len__(self):
        return self.total_items

    def __getitem__(self, idx):
        if idx >= self.total_items:
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {self.total_items}")
        file_idx = idx // len(self.start_positions)  # Which file (0 to N-1)
        depth_idx = idx % len(self.start_positions)  # Which depth slice
        depth_start = self.start_positions[depth_idx]
        depth_end = depth_start + self.volume_depth
        
        slice_data = self.data[file_idx, :, depth_start:depth_end, :, :].clone()  # [4, vol_depth, h, w]
        slice_mask = self.masks[file_idx, :, depth_start:depth_end, :, :].clone()  # [1, vol_depth, h, w]
        
        if self.augment and self.transforms:
            target_shape = [1, slice_mask.shape[-3], slice_mask.shape[-2], slice_mask.shape[-1]]
            while slice_mask.dim() < 4:
                slice_mask = slice_mask.unsqueeze(0)
            if slice_mask.shape[0] > 1:
                slice_mask = slice_mask[0:1]
            slice_mask_expanded = slice_mask.repeat(4, 1, 1, 1)  # [4, vol_depth, h, w]
            combined = torch.cat((slice_data, slice_mask_expanded), dim=0)  # [8, vol_depth, h, w]
            transformed = self.transforms(combined)
            slice_data = transformed[:4]  # [4, vol_depth, h, w]
            slice_mask = transformed[4:5]  # [1, vol_depth, h, w]
        
        return slice_data, slice_mask

# Step 4: Loss Functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

def calculate_metrics(predictions, targets, threshold=0.5):
    predictions = (torch.sigmoid(predictions) > threshold).float()
    targets = targets.float()
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    tp = (predictions * targets).sum().item()
    fp = (predictions * (1 - targets)).sum().item()
    fn = ((1 - predictions) * targets).sum().item()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return {"dice": dice, "iou": iou}

# Step 5: Training Loop
def training_loop(model, params):
    epochs = params["epochs"]
    loss_func = params["f_loss"]
    optimizer = params["optimizer"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]
    os.makedirs(os.path.dirname(weight_path) or '.', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")
    best_val_dice = 0.0
    patience, wait = 25, 0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
    global mri_tensor, mask_tensor
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True)
        for batch in train_bar:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = loss_func(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                loss.backward()
                optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_dl)
        history["train_loss"].append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0.0
        val_metrics = {"dice": 0.0, "iou": 0.0}
        val_bar = tqdm(val_dl, desc=f"Epoch {epoch+1}/{epochs} [Valid]", leave=True)
        with torch.no_grad():
            for batch in val_bar:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                total_val_loss += loss.item()
                batch_metrics = calculate_metrics(outputs, targets)
                for k, v in batch_metrics.items():
                    val_metrics[k] += v
                val_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_val_loss = total_val_loss / len(val_dl)
        for k in val_metrics:
            val_metrics[k] /= len(val_dl)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        lr_scheduler.step(avg_val_loss)
        
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save({'model_state_dict': model.state_dict(), 'dice': best_val_dice}, weight_path)
            logger.info(f"✓ Model saved with Dice: {best_val_dice:.4f}")
            wait = 0
        else:
            wait += 1
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        if wait >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}. Best Dice: {best_val_dice:.4f}")
            break
        
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                model.eval()
                segmented_data = segment_mri(mri_tensor, model, device)  # Shape: [N, D, H, W]
                original_data = mri_tensor.cpu().numpy()  # Shape: [N, 4, D, H, W]
                true_mask = mask_tensor.cpu().numpy()  # Shape: [N, 1, D, H, W]
                depth = original_data.shape[2]  # Depth dimension
                middle_idx = depth // 2
                slice_indices = [middle_idx - 4, middle_idx, middle_idx + 4]
                slice_indices = [idx for idx in slice_indices if 0 <= idx < depth]
                # Pass the entire batch and specify the number of files
                visualize_results(original_data, true_mask, segmented_data,
                                 num_files=mri_tensor.shape[0],
                                 slice_indices=slice_indices,
                                 save_path=os.path.join(os.path.dirname(weight_path), f'segmentation_epoch_{epoch+1}.png'))
    
    return history

# Step 6: Visualization for Multiple Files
def visualize_results(mri_data, true_mask, predicted_mask, num_files=1, slice_indices=None, save_path=None):
    """
    Visualize MRI, true mask, and predicted mask for multiple files.

    Args:
        mri_data (numpy.ndarray): Shape [N, 4, D, H, W], MRI data for N files.
        true_mask (numpy.ndarray): Shape [N, 1, D, H, W], true masks for N files.
        predicted_mask (numpy.ndarray): Shape [N, D, H, W], predicted masks for N files.
        num_files (int): Number of files to visualize.
        slice_indices (list): Specific depth indices to visualize; if None, select automatically.
        save_path (str): Base path for saving visualizations; will append file index.
    """
    # Ensure inputs are numpy arrays
    mri_data = np.array(mri_data)
    true_mask = np.array(true_mask)
    predicted_mask = np.array(predicted_mask)

    # Remove batch dimension for true_mask if present
    if true_mask.shape[1] == 1:
        true_mask = true_mask.squeeze(1)  # [N, D, H, W]

    # Loop over each file
    for file_idx in range(min(num_files, mri_data.shape[0])):
        logger.info(f"Visualizing results for file {file_idx}")

        # Extract data for the current file
        mri_display = mri_data[file_idx]  # [4, D, H, W]
        true_mask_display = true_mask[file_idx]  # [D, H, W]
        pred_mask_display = predicted_mask[file_idx]  # [D, H, W]

        # Select the first channel for MRI visualization
        if mri_display.shape[0] == 4:  # Assuming 4 channels
            mri_display = mri_display[0]  # [D, H, W]

        depth = mri_display.shape[0]

        # Find slices with sufficient non-zero data
        non_zero_slices = [idx for idx in range(depth) if np.sum(mri_display[idx] > 0) > 100]
        if not non_zero_slices:
            logger.warning(f"No slices with meaningful MRI data found for file {file_idx}! Using middle slice.")
            non_zero_slices = [depth // 2]

        # Select slices to visualize
        if slice_indices is None:
            middle_idx = len(non_zero_slices) // 2
            selected_indices = [
                non_zero_slices[max(0, middle_idx - 2)],
                non_zero_slices[middle_idx],
                non_zero_slices[min(len(non_zero_slices) - 1, middle_idx + 2)]
            ]
        else:
            selected_indices = [idx for idx in slice_indices if idx in non_zero_slices]
            if not selected_indices:
                logger.warning(f"Provided slice_indices do not contain meaningful data for file {file_idx}! Using defaults.")
                middle_idx = len(non_zero_slices) // 2
                selected_indices = [
                    non_zero_slices[max(0, middle_idx - 2)],
                    non_zero_slices[middle_idx],
                    non_zero_slices[min(len(non_zero_slices) - 1, middle_idx + 2)]
                ]

        n_slices = len(selected_indices)
        fig, axes = plt.subplots(3, n_slices, figsize=(n_slices * 4, 12))
        if n_slices == 1:
            axes = axes.reshape(3, 1)

        for i, slice_idx in enumerate(selected_indices):
            mri_slice = mri_display[slice_idx]
            vmin, vmax = np.percentile(mri_slice[mri_slice > 0], [1, 99]) if np.any(mri_slice > 0) else (0, 1)
            axes[0, i].imshow(mri_slice, cmap='gray', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f'File {file_idx} MRI Slice {slice_idx}')
            axes[0, i].axis('off')

            mask_slice = true_mask_display[slice_idx].astype(np.float32)
            axes[1, i].imshow(mask_slice, cmap='viridis', vmin=0, vmax=1)
            axes[1, i].set_title(f'File {file_idx} True Mask {slice_idx}')
            axes[1, i].axis('off')

            pred_slice = pred_mask_display[slice_idx].astype(np.float32)
            axes[2, i].imshow(pred_slice, cmap='viridis', vmin=0, vmax=1)
            axes[2, i].set_title(f'File {file_idx} Predicted Mask {slice_idx}')
            axes[2, i].axis('off')

        plt.tight_layout()

        # Save the plot for this file
        if save_path:
            file_save_path = save_path.replace('.png', f'_file_{file_idx}.png')
            plt.savefig(file_save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization for file {file_idx} saved to {file_save_path}")

        plt.show()
        plt.close(fig)  # Close the figure to free memory

# Step 7: DataLoader Creation
def create_dataloader(mri_tensor, mask_tensor, batch_size=4, volume_depth=16, overlap=4):
    train_dataset = MRIDataset(mri_tensor, mask_tensor, volume_depth, overlap, augment=True)
    val_dataset = MRIDataset(mri_tensor, mask_tensor, volume_depth, overlap, augment=False)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    return train_loader, val_loader

# Step 8: Segmentation and Post-Processing
def segment_mri(mri_tensor, model, device=None, threshold=0.5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    num_files, _, depth, height, width = mri_tensor.shape  # Extract number of files (e.g., 5)
    volume_depth = 16
    output_tensor = torch.zeros((num_files, 1, depth, height, width), device=device)  # Match num_files
    count_tensor = torch.zeros((num_files, 1, depth, height, width), device=device)  # Match num_files
    
    with torch.no_grad():
        for start_idx in range(0, depth - volume_depth + 1, volume_depth // 2):
            end_idx = min(start_idx + volume_depth, depth)
            if end_idx - start_idx < volume_depth:
                start_idx = end_idx - volume_depth
            # Process the full batch of files
            volume = mri_tensor[:, :, start_idx:end_idx, :, :].to(device)  # Shape: [num_files, 4, vol_depth, H, W]
            pred = torch.sigmoid(model(volume))  # Shape: [num_files, 1, vol_depth, H, W]
            output_tensor[:, :, start_idx:end_idx, :, :] += pred
            count_tensor[:, :, start_idx:end_idx, :, :] += 1
    
    output_tensor = output_tensor / count_tensor  # Average overlapping predictions
    segmented_data = output_tensor.cpu().numpy()  # Shape: [num_files, 1, D, H, W]
    segmented_data = np.array([post_process_mask(seg.squeeze(0), min_size=50) for seg in segmented_data])  # Process each file
    return (segmented_data > threshold).astype(np.float32)  # Shape: [num_files, D, H, W]

def post_process_mask(mask, min_size=50):
    labeled, num_features = label(mask > 0.5)
    for i in range(1, num_features + 1):
        if (labeled == i).sum() < min_size:
            mask[labeled == i] = 0
    return mask

# Step 9: Save Segmented Data
def save_segmented_data(segmented_data, affine, output_path):
    segmented_img = nib.Nifti1Image(segmented_data, affine)
    nib.save(segmented_img, output_path)
    logger.info(f"Segmented data saved to {output_path}")

def get_paired_file_paths(images_dir, labels_dir):
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz') and not f.startswith('.')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz') and not f.startswith('.')])
    paired_paths = []
    for img_file in image_files:
        base_name = img_file.split('.')[0]
        mask_file = f"{base_name}.nii.gz"
        if mask_file in label_files:
            paired_paths.append((os.path.join(images_dir, img_file), os.path.join(labels_dir, mask_file)))
        else:
            logger.warning(f"No matching mask found for {img_file}")
    logger.info(f"Found {len(paired_paths)} paired MRI and mask files")
    return paired_paths

# Main Function
def main():
    global mri_tensor, mask_tensor
    base_path = os.path.expanduser("~/Downloads/Task01_BrainTumour (1)/Task01_BrainTumour/")
    images_dir = os.path.join(base_path, "imagesTr")
    labels_dir = os.path.join(base_path, "labelsTr")
    output_dir = os.path.join(base_path, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    paired_paths = get_paired_file_paths(images_dir, labels_dir)
    if not paired_paths:
        raise FileNotFoundError("No valid paired MRI and mask files found in the directories")
    
    num_files_to_process = min(5, len(paired_paths))
    logger.info(f"Processing {num_files_to_process} files")
    
    all_mri_tensors = []
    all_mask_tensors = []
    reference_affine = None
    
    for idx, (mri_file_path, mask_file_path) in enumerate(paired_paths[:num_files_to_process]):
        logger.info(f"Processing file pair {idx + 1}/{num_files_to_process}: {os.path.basename(mri_file_path)}")
        mri_tensor, mri_scan = load_and_preprocess_mri(mri_file_path, is_mask=False)
        if reference_affine is None:
            reference_affine = mri_scan.affine
        _, _, depth, height, width = mri_tensor.shape
        mask_tensor, _ = load_and_preprocess_mri(mask_file_path, is_mask=True, target_shape=(depth, height, width))
        all_mri_tensors.append(mri_tensor)
        all_mask_tensors.append(mask_tensor)
    
    mri_tensor = torch.cat(all_mri_tensors, dim=0)  # [N, 4, D, H, W]
    mask_tensor = torch.cat(all_mask_tensors, dim=0)  # [N, 1, D, H, W]
    logger.info(f"Combined MRI tensor shape: {mri_tensor.shape}, Mask tensor shape: {mask_tensor.shape}")
    
    train_dl, val_dl = create_dataloader(mri_tensor, mask_tensor, batch_size=4, volume_depth=16, overlap=4)
    
    model = UNet3D(in_channels=4, out_channels=1, features=[16, 32, 64, 128])
    loss_func = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    def combined_loss(pred, target):
        bce = loss_func(pred, target)
        dice = dice_loss(pred, target)
        focal = focal_loss(pred, target)
        return 0.2 * bce + 0.5 * dice + 0.3 * focal
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)
    
    model_path = os.path.join(output_dir, 'model_weights.pth')
    params = {
        "epochs": 200,
        "f_loss": combined_loss,
        "optimizer": optimizer,
        "train": train_dl,
        "val": val_dl,
        "lr_change": lr_scheduler,
        "weight_path": model_path
    }
    
    logger.info("Starting training")
    history = training_loop(model, params)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmented_data = segment_mri(mri_tensor, model, device)  # Shape: [N, D, H, W]
    
    # Visualize results for all files
    original_data = mri_tensor.cpu().numpy()  # Shape: [N, 4, D, H, W]
    true_mask = mask_tensor.cpu().numpy()  # Shape: [N, 1, D, H, W]
    depth = original_data.shape[2]
    middle_slice = depth // 2
    slice_indices = [middle_slice - 4, middle_slice, middle_slice + 4]
    slice_indices = [idx for idx in slice_indices if 0 <= idx < depth]
    
    visualize_results(original_data, true_mask, segmented_data,
                     num_files=num_files_to_process,
                     slice_indices=slice_indices,
                     save_path=os.path.join(output_dir, 'segmentation_results.png'))
    
    # Save segmented data for all files
    for i in range(num_files_to_process):
        output_file_path = os.path.join(output_dir, f'segmented_mri_scan_{i}.nii.gz')
        save_segmented_data(segmented_data[i], reference_affine, output_file_path)

if __name__ == "__main__":
    main()