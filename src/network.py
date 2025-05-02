# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1Score 
import src.config as cfg
from src.util import show_setting

class MyNetwork(AlexNet): # AlexNet-Modified_1
    def __init__(self, num_classes: int = 200):
        super().__init__()

        # 1) 입력 크기 64×64에 맞춰 feature extractor 재정의
        #    - Conv + ReLU ×2 → MaxPool 순으로 축소 (64→32→16→8)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 64×64×3 → 64×64×64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # → 32×32×64

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),# → 32×32×192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                     # → 16×16×192

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),# → 16×16×384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),# → 16×16×256
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),# → 16×16×256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                     # → 8×8×256
        )

        # 2) classifier도 8×8→linear(256*8*8→4096)로 수정
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # (B, 256*8*8)
        x = self.classifier(x)
        return x

class MyNetwork(AlexNet): # AlexNet-Modified_2
    def __init__(self, num_classes: int = 200):
        super().__init__()

        # 1) 입력 크기 64x64 및 Batch Normalization 적용하여 feature extractor 재정의
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),      # 64x64x3 -> 64x64x64
            nn.BatchNorm2d(64), # BatchNorm 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # -> 32x32x64

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),    # -> 32x32x192
            nn.BatchNorm2d(192), # BatchNorm 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # -> 16x16x192

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),   # -> 16x16x384
            nn.BatchNorm2d(384), # BatchNorm 추가
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),   # -> 16x16x256
            nn.BatchNorm2d(256), # BatchNorm 추가
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),   # -> 16x16x256
            nn.BatchNorm2d(256), # BatchNorm 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # -> 8x8x256
        )

        # 2) Classifier는 이전과 동일 (입력 크기 256*8*8)
        #    (BatchNorm은 feature map 크기를 바꾸지 않으므로 수정 필요 없음)
        #    다만, 성능 향상을 위해 Classifier 내부에도 BatchNorm을 적용하거나
        #    구조를 변경하는 것을 고려해볼 수 있습니다 (여기서는 일단 유지).
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # (B, 256*8*8)
        x = self.classifier(x)
        return x

class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score(num_classes=num_classes, average='macro') # 추가 

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1score(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy, 'f1/train': f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1score(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy, 'f1/val': f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
