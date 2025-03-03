import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule
from typing import Dict, Optional
from torchmetrics import JaccardIndex

MODEL_REGISTRY = {
    "unet": smp.Unet,
    "deeplabv3": smp.DeepLabV3,
    "pspnet": smp.PSPNet,
    "manet": smp.MAnet,
    "pan": smp.PAN
}

class SegmentationModel(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 7,
        img_size: int = 512,
        learning_rate: float = 1e-3,
        class_weights: Optional[Dict[int, float]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model with pretrained encoder
        self.model = MODEL_REGISTRY[model_name](
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        
        # Initialize metrics
        self.train_iou = JaccardIndex(
            task='multiclass',
            num_classes=num_classes,
            average='macro'
        )
        self.val_iou = JaccardIndex(
            task='multiclass',
            num_classes=num_classes,
            average='macro'
        )
        self.train_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        
        # Track inference time and memory
        self.inference_times = []
        self.memory_usage = []
        
        # Class weighting
        self.class_weights = None
        if class_weights:
            weight_tensor = torch.tensor([class_weights[i] for i in range(num_classes)])
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.long())
        
        train_iou = self.train_iou(y_hat.softmax(dim=1), y)
        train_loss = self.train_loss(y_hat, y.long())
        
        self.log("train_loss", train_loss, on_step=True, on_epoch=True)
        self.log("train_iou", train_iou, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.long())
        
        val_iou = self.val_iou(y_hat.softmax(dim=1), y)
        val_loss = self.val_loss(y_hat, y.long())
        
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        self.log("val_iou", val_iou, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        return [optimizer], [scheduler]

    def benchmark(self, dataloader) -> Dict[str, float]:
        """Run standardized benchmarking metrics"""
        self.eval()
        device = next(self.parameters()).device
        ious = []
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                
                # Timing and memory measurement
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                torch.cuda.synchronize()
                start_event.record()
                outputs = self(x)
                end_event.record()
                torch.cuda.synchronize()
                
                # Calculate metrics
                preds = torch.argmax(outputs, dim=1)
                ious.append(self.val_iou(preds, y))
                
                # Record resources
                self.inference_times.append(start_event.elapsed_time(end_event))
                if torch.cuda.is_available():
                    self.memory_usage.append(torch.cuda.max_memory_allocated(device))

        return {
            "mean_iou": torch.mean(torch.tensor(ious)),
            "mean_inference_time": torch.mean(torch.tensor(self.inference_times)),
            "peak_memory_usage": torch.max(torch.tensor(self.memory_usage)) if self.memory_usage else 0
        }
