from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int, average: str = 'macro', eps: float = 1e-9):
        super().__init__()
        self.num_classes = num_classes
        self.average     = average
        self.eps         = eps

        # 클래스별 TP, FP, FN을 추적
        self.add_state('tp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds: (B, C), target: (B,)
        """
        preds_label = torch.argmax(preds, dim=1)
        # 각 클래스마다 TP/FP/FN 누적
        for cls in range(self.num_classes):
            cls_pred = (preds_label == cls)
            cls_true = (target     == cls)
            self.tp[cls] += torch.sum( cls_pred &  cls_true)
            self.fp[cls] += torch.sum( cls_pred & ~cls_true)
            self.fn[cls] += torch.sum(~cls_pred &  cls_true)

    def compute(self):
        # precision, recall, f1 계산
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall    = self.tp / (self.tp + self.fn + self.eps)
        f1        = 2 * precision * recall / (precision + recall + self.eps)

        if self.average == 'macro':
            return f1.mean()
        elif self.average == 'none':
            return f1      # 클래스별 F1을 모두 리턴
        else:
            raise ValueError(f"Unknown average: {self.average}")

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds_label = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds_label.shape != target.shape:
            raise ValueError(f"Shape mismatch: {preds_label.shape} vs {target.shape}")

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds_label == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
