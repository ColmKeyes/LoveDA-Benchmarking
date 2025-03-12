 LoveDA Semantic Segmentation Benchmark

## Project Documentation

For architectural overview and component relationships, see [ARCHITECTURE.md](ARCHITECTURE.md)  
Detailed technical specifications and implementation notes available in [DOCS.md](DOCS.md)

## High-Level Objectives
- [x] Build benchmarking scripts for LoveDA dataset
- [✔] Implement comprehensive comparison of EO models (in testing)
- [✔] Finalize visualization system (v1 implemented)

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
- CI/CD pipeline integration

## Current Implementation Status
**Latest Benchmarks**  
See [results/README.md](results/README.md) for latest metrics

**Recent Changes**  
- Added UNet++ and PSPNet architectures  
- Integrated mixed precision training support  
- Automated benchmark report comparison system  
- CI/CD readiness checks implemented  

## Completed in Testing Phase
1. Data augmentation pipeline (v2.1 implemented)
2. Prediction visualization hooks (see /src/lovebench/visualization.py)
3. Extended metrics suite (Dice, IoU, Precision/Recall)

## Implementation Questions
✅ Resolved:
1. Dataset split ratios: 70-15-15 train/val/test
2. Class weights calculated globally
3. Cloud detection handled at dataset level
4. CUDA visibility confirmed through environment validation checks
5. CLI working directory standardized to project root

## Usage

**Environment**: Torchgeo_Benchmarks (Python 3.8)  
**Working Directory**: Execute from project root (/home/colm-the-conjurer/VSCode/workspace/LoveDA-Benchmarking)

```bash
# Run full benchmark suite
python -m lovebench.cli \
  --models deeplabv3 fcn_resnet50 lraspp_mobilenet unetplusplus pspnet \
  --data-root ./data/LoveDA \
  --output benchmarks
```

## Dependencies
- Python 3.8+
- PyTorch 2.0.1+
- TorchGeo 0.5.2+
- PyTorch Lightning 2.1.0+
- CUDA 11.8+
