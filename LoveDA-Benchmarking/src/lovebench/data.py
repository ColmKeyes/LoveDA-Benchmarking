from torchgeo.datasets import LoveDA
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple
import torch
import numpy as np

def load_data(
    root_path: str,
    split: str = "test",
    img_size: int = 512,
    batch_size: int = 32,
    augment: bool = False
) -> Tuple[DataLoader, int]:
    """Load LoveDA dataset with transformations
    
    Args:
        root_path: Path to dataset root directory
        split: One of 'train', 'val', 'test'
        img_size: Target size for image resizing
        batch_size: Batch size for DataLoader
        augment: Apply training augmentations
        
    Returns:
        Tuple of (DataLoader, number_of_classes)
    """
    # Define transformations
    if augment:
        transform = transforms.Compose([
            transforms.Lambda(lambda sample: {
                'image': transforms.functional.resize(
                    sample['image'], 
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                'mask': transforms.functional.resize(
                    sample['mask'], 
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.NEAREST
                )
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
                'image': transforms.functional.adjust_brightness(
                    transforms.functional.adjust_contrast(
                        transforms.functional.adjust_saturation(
                            transforms.functional.adjust_hue(
                                sample['image'],
                                hue_factor=0.1
                            ),
                            saturation_factor=0.2
                        ),
                        contrast_factor=0.2
                    ),
                    brightness_factor=0.2
                ),
                'mask': sample['mask']
            }),
            transforms.Lambda(lambda sample: {
                'image': transforms.functional.rotate(
                    sample['image'], 
                    degrees=15
                ),
                'mask': transforms.functional.rotate(
                    sample['mask'], 
                    degrees=15
                )
            }),
            transforms.Lambda(lambda sample: {
                'image': transforms.functional.to_tensor(sample['image']),
                'mask': torch.as_tensor(np.array(sample['mask']), dtype=torch.long)
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
                'image': transforms.functional.resize(
                    sample['image'],
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                'mask': transforms.functional.resize(
                    sample['mask'],
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.NEAREST
                )
            }),
            transforms.Lambda(lambda sample: {
                'image': transforms.functional.to_tensor(sample['image']),
                'mask': torch.as_tensor(np.array(sample['mask']), dtype=torch.long)
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
        download=True,
        checksum=True
    )
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=(split == "train")
    )
    
    return loader, 7  # LoveDA has 7 classes

def get_class_weights(root_path: str) -> dict:
    """Calculate class weights for imbalance correction"""
    dataset = LoveDA(root=root_path, split="train", download=True)
    class_counts = torch.zeros(7)
    
    for sample in dataset:
        label = sample['mask']
        class_counts += torch.bincount(
            label.flatten(), 
            minlength=7
        )
        
    # Add epsilon to avoid division by zero
    class_weights = 1.0 / (class_counts / class_counts.sum() + 1e-6)
    return {i: weight.item() for i, weight in enumerate(class_weights.normalize())}
