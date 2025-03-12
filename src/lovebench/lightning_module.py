import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from .models import SegmentationModel, MODEL_REGISTRY

class LoveDABenchmarkModule(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int = 7, img_size: int = 512):
        super().__init__()
        self.model = SegmentationModel(
            model_name=model_name,
            num_classes=num_classes,
            img_size=img_size
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.iou_metric = JaccardIndex(num_classes=num_classes, task='multiclass')
        self.best_val_iou = 0.0
        self.val_iou = 0.0
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # Calculate IoU for training metrics
        pred = torch.argmax(outputs, dim=1)
        iou = self.iou_metric(pred, targets)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_iou', iou, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # Calculate IoU for validation metrics
        pred = torch.argmax(outputs, dim=1)
        iou = self.iou_metric(pred, targets)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # Calculate IoU for test metrics
        pred = torch.argmax(outputs, dim=1)
        iou = self.iou_metric(pred, targets)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_iou', iou, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        return [optimizer], [scheduler]

    def on_validation_epoch_end(self):
        if self.val_iou > self.best_val_iou:
            self.best_val_iou = self.val_iou
            torch.save(self.model.state_dict(), "best_model.pth")

    def benchmark(self, test_loader):
        """Run full benchmark with timing and memory monitoring"""
        self.eval()
        device = next(self.parameters()).device
        ious = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                # Time inference
                torch.cuda.synchronize()
                start_event.record()
                outputs = self(images)
                end_event.record()
                torch.cuda.synchronize()
                
                # Calculate metrics
                preds = torch.argmax(outputs, dim=1)
                ious.append(self.iou_metric(preds, targets))
                
                # Track memory usage
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)

        return {
            "mean_iou": torch.mean(torch.tensor(ious)),
            "mean_inference_time": start_event.elapsed_time(end_event) / len(test_loader),
            "peak_memory_usage": torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        }
