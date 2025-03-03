import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Dict
from .models import SegmentationModel, MODEL_REGISTRY
from .data import load_data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
import torch

def main():
    parser = argparse.ArgumentParser(description="Run LoveDA benchmarks")
    
    # Required arguments
    parser.add_argument("--models", nargs="+", 
                       choices=MODEL_REGISTRY.keys(),
                       required=True,
                       help="Models to benchmark")
    parser.add_argument("--data-root", type=str, required=True,
                       help="Path to LoveDA dataset")
    
    # Optional arguments
    parser.add_argument("--img-size", type=int, default=512,
                       help="Input image size (default: 512)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for benchmarking (default: 32)")
    parser.add_argument("--output", type=str, default="benchmarks",
                       help="Output directory for results (default: benchmarks)")
    
    args = parser.parse_args()

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    test_loader, _ = load_data(
        root_path=args.data_root,
        split="test",
        img_size=args.img_size,
        batch_size=args.batch_size,
        augment=False
    )

    results = []
    for model_name in args.models:
        print(f"\nBenchmarking {model_name}...")
        
        # Initialize model
        model = SegmentationModel(
            model_name=model_name,
            num_classes=7,
            img_size=args.img_size
        )
        
        # Configure trainer
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            callbacks=[RichProgressBar()],
            enable_checkpointing=False,
            logger=False
        )
        
        # Run validation for metrics
        trainer.validate(model, test_loader)
        
        # Collect benchmark metrics
        metrics = model.benchmark(test_loader)
        
        # Store results
        results.append({
            "Model": model_name,
            "Backbone": model_name.split("_")[-1],
            "Input Size": args.img_size,
            "mIoU": metrics["mean_iou"].item(),
            "Inference Time (ms)": metrics["mean_inference_time"],
            "Memory Usage (GB)": metrics["peak_memory_usage"] / 1e9,
            "Date": datetime.now().strftime("%Y-%m-%d")
        })
        
        # Update README after each model
        update_readme(results, output_dir)

    # Save full results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "benchmark_results.csv", index=False)
    print(f"\nSaved results to {output_dir}/benchmark_results.csv")

def update_readme(results: List[Dict], output_dir: Path):
    """Update README.md with latest benchmark results"""
    readme_path = output_dir / "README.md"
    
    # Create markdown table
    md_table = "| Model | Backbone | Input Size | mIoU | Inference Time (ms) | Memory Usage (GB) | Date |\n"
    md_table += "|-------|----------|------------|------|---------------------|-------------------|------|\n"
    
    for result in results:
        md_table += (
            f"| {result['Model']} | {result['Backbone']} | {result['Input Size']} | "
            f"{result['mIoU']:.2f} | {result['Inference Time (ms)']:.1f} | "
            f"{result['Memory Usage (GB)']:.1f} | {result['Date']} |\n"
        )
    
    # Write/update README
    with open(readme_path, "w") as f:
        f.write("# LoveDA Benchmark Results\n\n")
        f.write("## Latest Results\n")
        f.write(md_table)
        f.write("\n\n## Historical Results\n")
        f.write("See benchmark_results.csv for complete historical data")

if __name__ == "__main__":
    main()
