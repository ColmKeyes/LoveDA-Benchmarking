import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Dict, Any
from lovebench.models import SegmentationModel
from lovebench.data import load_data, get_class_weights

def run_benchmark(
    data_root: str,
    model_name: str = "unet",
    img_size: int = 512,
    batch_size: int = 32,
    output_dir: str = "results",
    class_weights: Optional[Dict[int, float]] = None,
    learning_rate: float = 1e-3,
    split: str = "train"
) -> Dict[str, Any]:
    """Run LoveDA benchmarking with configurable parameters
    
    Args:
        data_root: Path to dataset root directory
        model_name: Name of model architecture to use
        img_size: Input image size
        batch_size: Batch size for training/validation
        output_dir: Directory to save results
        class_weights: Optional class weights for loss function
        learning_rate: Learning rate for optimizer
        split: Dataset split to use ('train', 'val', 'test')
        
    Returns:
        Dictionary containing benchmark results
    """
    # Setup
    torch.set_float32_matmul_precision('high')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    test_loader, _ = load_data(
        root_path=data_root,
        split=split,
        img_size=img_size,
        batch_size=batch_size,
        augment=False
    )
    
    # Initialize model
    print(f"Benchmarking {model_name}...")
    model = SegmentationModel(
        model_name=model_name,
        num_classes=7,
        img_size=img_size,
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        enable_checkpointing=False,
        logger=False
    )
    
    # Run validation
    trainer.validate(model, test_loader)
    
    # Get benchmark metrics
    metrics = model.benchmark(test_loader)
    
    return metrics

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LoveDA benchmarks")
    parser.add_argument("--data-root", type=str, required=True,
                       help="Path to dataset root directory")
    parser.add_argument("--model", type=str, default="unet",
                       choices=["unet", "deeplabv3", "pspnet", "manet", "pan"],
                       help="Model architecture to benchmark")
    parser.add_argument("--img-size", type=int, default=512,
                       help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training/validation")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate for optimizer")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test"],
                       help="Dataset split to use")
    parser.add_argument("--use-class-weights", action="store_true",
                       help="Use class weights in loss function")
    
    args = parser.parse_args()
    
    # Get class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = get_class_weights(args.data_root)
    
    # Run benchmark
    metrics = run_benchmark(
        data_root=args.data_root,
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        class_weights=class_weights,
        learning_rate=args.learning_rate,
        split=args.split
    )
    
    # Print results
    print("\nBenchmark Results:")
    print("-----------------")
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
