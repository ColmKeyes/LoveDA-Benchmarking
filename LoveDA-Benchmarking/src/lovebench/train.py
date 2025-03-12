import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Dict, Any
from lovebench.models import SegmentationModel
from lovebench.data import load_data, get_class_weights

def train_model(
    data_root: str,
    model_name: str = "unet",
    img_size: int = 512,
    batch_size: int = 32,
    output_dir: str = "results",
    class_weights: Optional[Dict[int, float]] = None,
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """Train a model on the LoveDA dataset
    
    Args:
        data_root: Path to dataset root directory
        model_name: Name of model architecture to use
        img_size: Input image size
        batch_size: Batch size for training/validation
        output_dir: Directory to save results
        class_weights: Optional class weights for loss function
        learning_rate: Learning rate for optimizer
        max_epochs: Maximum number of training epochs
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        checkpoint_path: Optional path to load model checkpoint from
        
    Returns:
        Dictionary containing training results
    """
    # Setup
    torch.set_float32_matmul_precision('high')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    train_loader, _ = load_data(
        root_path=data_root,
        split="train",
        img_size=img_size,
        batch_size=batch_size,
        augment=True
    )
    
    val_loader, _ = load_data(
        root_path=data_root,
        split="val",
        img_size=img_size,
        batch_size=batch_size,
        augment=False
    )
    
    # Initialize model
    print(f"Training {model_name}...")
    model = SegmentationModel(
        model_name=model_name,
        num_classes=7,
        img_size=img_size,
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    
    # Setup callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename=f"{model_name}-{{epoch:02d}}-{{val_iou:.2f}}",
            monitor="val_iou",
            mode="max",
            save_top_k=1
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_iou",
            mode="max",
            patience=early_stopping_patience,
            verbose=True
        )
    ]
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=True
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Get best metrics
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_metrics = {
        "best_model_path": best_model_path,
        "best_val_iou": trainer.checkpoint_callback.best_model_score.item()
    }
    
    return best_metrics

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LoveDA model")
    parser.add_argument("--data-root", type=str, required=True,
                       help="Path to dataset root directory")
    parser.add_argument("--model", type=str, default="unet",
                       choices=["unet", "deeplabv3", "pspnet", "manet", "pan"],
                       help="Model architecture to train")
    parser.add_argument("--img-size", type=int, default=512,
                       help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training/validation")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate for optimizer")
    parser.add_argument("--max-epochs", type=int, default=100,
                       help="Maximum number of training epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                       help="Number of epochs to wait for improvement before stopping")
    parser.add_argument("--use-class-weights", action="store_true",
                       help="Use class weights in loss function")
    parser.add_argument("--checkpoint-path", type=str,
                       help="Optional path to load model checkpoint from")
    
    args = parser.parse_args()
    
    # Get class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = get_class_weights(args.data_root)
    
    # Train model
    best_metrics = train_model(
        data_root=args.data_root,
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        class_weights=class_weights,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_path=args.checkpoint_path
    )
    
    # Print results
    print("\nTraining Results:")
    print("----------------")
    for k, v in best_metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
