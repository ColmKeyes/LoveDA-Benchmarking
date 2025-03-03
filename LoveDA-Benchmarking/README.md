LoveDA Semantic Segmentation Benchmark

## Project Documentation

For architectural overview and component relationships, see [ARCHITECTURE.md](ARCHITECTURE.md)  
Detailed technical specifications and implementation notes available in [DOCS.md](DOCS.md)

## High-Level Objectives
- [x] Build benchmarking scripts for LoveDA dataset
- [ ] Implement comprehensive comparison of EO models
- [ ] Finalize visualization system

## Mid-Level Progress
✅ Completed:
- Model initialization & configuration (torchvision models)
- PyTorch Lightning integration
- CLI benchmarking interface
- Automated documentation updates

✅ Completed:
- Dataset ingestion & transformation pipeline
- Class weight integration

🔄 In Progress:
- Multi-GPU training support

## Current Implementation Status
**Latest Benchmarks**  
See [results/README.md](results/README.md) for latest metrics

**Recent Changes**  
- Added DeepLabV3, FCN-ResNet50, LR-ASPP MobileNetV3  
- Implemented memory/inference time tracking  
- Automated Markdown report generation  

## Pending Tasks
1. Finalize data augmentation pipeline
2. Add visualization hooks for prediction samples
3. Integrate additional metrics (Dice, Precision/Recall)

## Implementation Questions
✅ Resolved:
1. Dataset split ratios: 70-15-15 train/val/test
2. Class weights calculated globally
3. Cloud detection handled at dataset level

🆕 New Questions:
1. Verify CUDA visibility in Torchgeo_Benchmarks environment
2. Confirm working directory for CLI execution

## Usage

**Environment**: Torchgeo_Benchmarks (Python 3.8)  
**Working Directory**: Execute from project root (/home/colm-the-conjurer/VSCode/workspace/LoveDA-Benchmarking)

```bash
# Run full benchmark suite
python -m lovebench.cli \
  --models deeplabv3 fcn_resnet50 lraspp_mobilenet \
  --data-root ./data/LoveDA \
  --output benchmarks
```

## Dependencies
- Python 3.8+
- PyTorch 1.12+
- TorchGeo 0.4+
- PyTorch Lightning 2.0+
