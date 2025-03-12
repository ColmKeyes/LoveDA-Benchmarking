# Model Architecture Details

## Available Models
The following models are available for use in this project:

### 1. UNet
- **Source**: PyTorch implementation
- **Parameters**: ~40M (depending on variant)
- **Backbone**: ResNet-50 (pre-trained on ImageNet)
- **Encoder Head**: Double convolution with skip connections

### 2. Deeplabv3
- **Source**: PyTorch implementation
- **Parameters**: ~69M
- **Backbone**: ResNet-101 (pre-trained on ImageNet)
- **Encoder Head**: ASPP (Atrous Spatial Pyramid Pooling)

### 3. PSPNet
- **Source**: PyTorch implementation
- **Parameters**: ~58M
- **Backbone**: ResNet-50 (pre-trained on ImageNet)
- **Encoder Head**: PSP Module (Pyramid Scene Parsing)

### 4. MANet
- **Source**: Modified from original paper
- **Parameters**: ~30M
- **Backbone**: MobileNet-V2 (pre-trained on ImageNet)
- **Encoder Head**: Light-weighted attention module

### 5. PAN
- **Source**: PyTorch implementation
- **Parameters**: ~140M
- **Backbone**: ResNet-101 (pre-trained on ImageNet)
- **Encoder Head**: Pyramid Attention Module

Each model is configured for semantic segmentation tasks and can be loaded with pre-trained weights or trained from scratch.
