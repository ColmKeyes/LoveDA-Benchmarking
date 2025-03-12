import torch
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
from datetime import datetime
from lovebench.models import SegmentationModel
from lovebench.data import load_data


def main():
    """Simple script to run LoveDA benchmarks"""
    # Fixed configuration
    data_root = "/home/colm-the-conjurer/Data/datasets/LoveDA"
    model_name = "unet"
    img_size = 512
    batch_size = 32
    output_dir = "results"

    # Setup
    torch.set_float32_matmul_precision('high')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load test data
    print("Loading dataset...")
    test_loader, _ = load_data(
        root_path=data_root,
        split="train",
        img_size=img_size,
        batch_size=batch_size,
        augment=False
    )

    # Initialize model and trainer
    print(f"Benchmarking {model_name}...")
    model = SegmentationModel(
        model_name=model_name,
        num_classes=7,
        img_size=img_size
    )

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

    # Save results
    results = {
        "Model": model_name,
        "Input Size": img_size,
        "mIoU": metrics["mean_iou"].item(),
        "Date": datetime.now().strftime("%Y-%m-%d")
    }

    # Add CUDA metrics if available
    if "mean_inference_time" in metrics:
        results["Inference Time (ms)"] = metrics["mean_inference_time"].item()
    if "peak_memory_usage" in metrics:
        results["Memory Usage (GB)"] = metrics["peak_memory_usage"].item() / 1e9

    df = pd.DataFrame([results])
    results_path = Path(output_dir) / "benchmark_results.csv"
    df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
