from torchgeo.datasets import LoveDA
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple
import torch
import numpy as np

def load_data(
    root_path: str,
    split: str = "train",
    img_size: int = 512,
    batch_size: int = 32,
    augment: bool = False
) -> Tuple[DataLoader, int]:
    """Load LoveDA dataset with transformations
    
    Args:
        root_path: Path to dataset root directory
        split: One of 'train', 'val', 'test'
        img_size: Target size for image resizing (applied to both dimensions)
        batch_size: Batch size for DataLoader
        augment: Apply training augmentations
        
    Returns:
        Tuple of (DataLoader, number_of_classes)
    """
    def ensure_tensor_format(image):
        """Ensure image is a properly formatted tensor with channels first"""
        if isinstance(image, torch.Tensor):
            # If tensor is in HWC format, convert to CHW
            if image.shape[-1] == 3:
                return image.permute(2, 0, 1)
            return image
        # If PIL Image or ndarray, convert to tensor (CHW format)
        return transforms.functional.to_tensor(image)

    def process_mask(mask):
        """Process mask to ensure valid class indices"""
        mask_array = np.array(mask)
        # Ensure values are in valid range [0, 6]
        mask_array = np.clip(mask_array, 0, 6)
        return torch.from_numpy(mask_array).long()

    # Define transformations
    if augment:
        transform = transforms.Compose([
            transforms.Lambda(lambda sample: {
                'image': ensure_tensor_format(sample['image']),
                'mask': process_mask(sample['mask'])
            }),
            transforms.Lambda(lambda sample: {
                'image': transforms.functional.resize(
                    sample['image'],
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                'mask': transforms.functional.resize(
                    sample['mask'].unsqueeze(0),
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.NEAREST
                ).squeeze(0)
            }),
            transforms.Lambda(lambda sample: {
                'image': transforms.functional.hflip(sample['image']) 
                        if torch.rand(1) < 0.5 else sample['image'],
                'mask': transforms.functional.hflip(sample['mask']) 
                       if torch.rand(1) < 0.5 else sample['mask']
            }),
            transforms.Lambda(lambda sample: {
                'image': transforms.functional.vflip(sample['image']) 
                        if torch.rand(1) < 0.5 else sample['image'],
                'mask': transforms.functional.vflip(sample['mask']) 
                       if torch.rand(1) < 0.5 else sample['mask']
            }),
            transforms.Lambda(lambda sample: {
                'image': transforms.functional.normalize(
                    sample['image'],
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                'mask': sample['mask']
            })
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda sample: {
                'image': ensure_tensor_format(sample['image']),
                'mask': process_mask(sample['mask'])
            }),
            transforms.Lambda(lambda sample: {
                'image': transforms.functional.resize(
                    sample['image'],
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                'mask': transforms.functional.resize(
                    sample['mask'].unsqueeze(0),
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.NEAREST
                ).squeeze(0)
            }),
            transforms.Lambda(lambda sample: {
                'image': transforms.functional.normalize(
                    sample['image'],
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                'mask': sample['mask']
            })
        ])

    # Load dataset
    dataset = LoveDA(
        root=root_path,
        split=split,
        transforms=transform,
        download=False,
        checksum=True
    )
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=(split.lower() == "train")
    )
    
    return loader, 7  # LoveDA has 7 classes

def get_class_weights(root_path: str) -> dict:
    """Calculate class weights for imbalance correction"""
    dataset = LoveDA(root=root_path, split="train", download=True)
    class_counts = torch.zeros(7)
    
    for sample in dataset:
        label = sample['mask']
        class_counts += torch.bincount(
            torch.as_tensor(np.array(label)).flatten(), 
            minlength=7
        )
        
    # Add epsilon to avoid division by zero
    class_weights = 1.0 / (class_counts / class_counts.sum() + 1e-6)
    return {i: weight.item() for i, weight in enumerate(class_weights)}
