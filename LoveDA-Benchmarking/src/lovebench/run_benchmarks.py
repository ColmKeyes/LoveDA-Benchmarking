import torch
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
from datetime import datetime
from lovebench.models import SegmentationModel, MODEL_REGISTRY
from lovebench.data import load_data





def main():
    """Run benchmarks for all registered models"""
    # Configuration
    data_root = "/home/colm-the-conjurer/Data/datasets/LoveDA"
    img_size = 512
    batch_size = 32
    output_dir = "results"
    
    # Setup
    torch.set_float32_matmul_precision('high')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load test data once
    print("Loading dataset...")
    test_loader, _ = load_data(
        root_path=data_root,
        split="train",
        img_size=img_size,
        batch_size=batch_size,
        augment=False
    )
    
    results = []
    
    # Benchmark each model
    for model_name in MODEL_REGISTRY:
        print(f"\nBenchmarking {model_name}...")
        
        # Initialize model
        model = SegmentationModel(
            model_name=model_name,
            num_classes=7,
            img_size=img_size
        )
        
        # Initialize trainer
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            enable_checkpointing=False,
            logger=False
        )
        
        # Validate first to initialize model properly
        trainer.validate(model, test_loader)
        
    ########################################
        # # Run benchmark
        # metrics = model.benchmark(test_loader)
        #
        # # Collect results
        # result_entry = {
        #     "Model": model_name,
        #     "Input Size": img_size,
        #     "mIoU": metrics["mean_iou"].item(),
        #     "Date": datetime.now().strftime("%Y-%m-%d")
        # }
        #
        # # Add CUDA metrics if available
        # if "mean_inference_time" in metrics:
        #     result_entry["Inference Time (ms)"] = metrics["mean_inference_time"].item()
        # if "peak_memory_usage" in metrics:
        #     result_entry["Memory Usage (GB)"] = metrics["peak_memory_usage"].item() / 1e9
        #
        # results.append(result_entry)
        # print(f"Completed {model_name} benchmark")
        #
    # Save all results
    # df = pd.DataFrame(results)
    # results_path = Path(output_dir) / "benchmark_results.csv"
    # df.to_csv(results_path, index=False)
    # print(f"\nSaved all results to {results_path}")
    ########################################

if __name__ == "__main__":
    main()
