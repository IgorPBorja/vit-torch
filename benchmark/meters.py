import typing as T
import numpy as np
import torch


#  class BinaryMeter:
#      """
#      Meter class for calculating metrics for binary classification
#      """
#
#      def __init__(self):
#          self.TP = 0
#          self.TN = 0
#          self.FP = 0
#          self.FN = 0
#          self.total = 0
#
#      def update(self, pred: torch.Tensor, label: torch.Tensor, as_probabilities=False, threshold=0.5):
#          if pred.shape != label.shape:
#              raise ValueError(f"Shape of predictions {pred.shape} does not match label shape {label.shape}")
#          if as_probabilities:
#              pred = pred > threshold
#          self.TP += ((pred == 1) & (label == 1)).sum()
#          self.TN += ((pred == 0) & (label == 0)).sum()
#          self.FP += ((pred == 1) & (label == 0)).sum()
#          self.FN += ((pred == 0) & (label == 1)).sum()
#          self.total += pred.size
#
#      def clear(self):
#          self.TP = 0
#          self.TN = 0
#          self.FP = 0
#          self.FN = 0
#          self.total = 0
#
#      def confusion_matrix(self):
#          return torch.tensor([[self.TP, self.FP], [self.FN, self.TN]])
#
#      def calculate_metrics(self, clear=True):
#          metrics_dict = {
#              "accuracy": self.accuracy(),
#              "precision": self.precision(),
#              "recall": self.recall(),
#              "f1": self.f1()
#          }
#          if clear:
#              self.clear()
#          return metrics_dict
#
#      def accuracy(self):
#          return (self.TP + self.TN) / self.total
#
#      def precision(self):
#          return self.TP / (self.TP + self.FP)
#
#      def recall(self):
#          return self.TP / (self.TP + self.FN)
#
#      def f1(self):
#          return 2 * self.precision() * self.recall() / (self.precision() + self.recall())
#
#      def __repr__(self):
#          return f"Meter(TP={self.TP}, TN={self.TN}, FP={self.FP}, FN={self.FN}, total={self.total})"
#
#      def __str__(self, show_quantities=False):
#          s = ""
#          if show_quantities:
#              s += f"TP: {self.TP}\n"
#              s += f"TN: {self.TN}\n"
#              s += f"FP: {self.FP}\n"
#              s += f"FN: {self.FN}\n"
#              s += f"Total: {self.total}\n"
#          s += f"Accuracy: {self.accuracy()}\n"
#          s += f"Precision: {self.precision()}\n"
#          s += f"Recall: {self.recall()}\n"
#          s += f"F1: {self.f1()}"
#          return s


class MulticlassMeter:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.TP = torch.zeros(num_classes)
        self.TN = torch.zeros(num_classes)
        self.FP = torch.zeros(num_classes)
        self.FN = torch.zeros(num_classes)
        self._epsilon = torch.tensor(np.finfo(np.float32).eps)
        self.total = 0

    def update(self, pred: torch.Tensor, label: torch.Tensor, as_probabilities=True):
        if as_probabilities:
            pred = torch.argmax(pred, dim=-1)  # (B, C) -> (B,)
        for i in range(self.num_classes):
            self.TP[i] += ((pred == i) & (label == i)).sum()
            self.TN[i] += ((pred != i) & (label != i)).sum()
            self.FP[i] += ((pred == i) & (label != i)).sum()
            self.FN[i] += ((pred != i) & (label == i)).sum()
        self.total += len(pred)

    def clear(self):
        self.TP = torch.zeros(self.num_classes)
        self.TN = torch.zeros(self.num_classes)
        self.FP = torch.zeros(self.num_classes)
        self.FN = torch.zeros(self.num_classes)
        self.total = 0

    def confusion_matrix(self):
        return torch.stack([self.TP, self.FP, self.FN, self.TN])

    def calculate_metrics(self, reduction='mean', clear=True, as_scalar=True) -> T.Dict[str, T.Union[float, torch.Tensor]]:
        if as_scalar and reduction not in ['mean', 'sum']:
            raise ValueError(f"Reduction {reduction} not recognized or incompatible with scalar output. Must be 'mean' or 'sum'")
        metrics_dict = {
            "top1": self.top1(),
            "accuracy": self.accuracy(reduction=reduction),
            "precision": self.precision(reduction=reduction),
            "recall": self.recall(reduction=reduction),
            "f1": self.f1(reduction=reduction)
        }
        if clear:
            self.clear()
        if as_scalar:
            return {k: v.item() for k, v in metrics_dict.items()}
        else:
            return metrics_dict

    def top1(self) -> torch.Tensor:
        return torch.sum(self.TP) / self.total

    def accuracy(self, reduction: str = 'mean') -> torch.Tensor:
        if reduction == 'mean':
            return ((self.TP + self.TN) / self.total).mean()
        elif reduction == 'sum':
            return ((self.TP + self.TN) / self.total).sum()
        elif reduction == 'none' or reduction is None:
            return (self.TP + self.TN) / self.total
        else:
            raise ValueError(f"Reduction {reduction} not recognized")

    def precision(self, reduction: str = 'mean') -> torch.Tensor:
        if reduction == 'mean':
            return (self.TP / (self.TP + self.FP + self._epsilon)).mean()
        elif reduction == 'sum':
            return (self.TP / (self.TP + self.FP + self._epsilon)).sum()
        elif reduction == 'none' or reduction is None:
            return self.TP / (self.TP + self.FP + self._epsilon)
        else:
            raise ValueError(f"Reduction {reduction} not recognized")

    def recall(self, reduction: str = 'mean') -> torch.Tensor:
        if reduction == 'mean':
            return (self.TP / (self.TP + self.FN + self._epsilon)).mean()
        elif reduction == 'sum':
            return (self.TP / (self.TP + self.FN + self._epsilon)).sum()
        elif reduction == 'none' or reduction is None:
            return self.TP / (self.TP + self.FN + self._epsilon)
        else:
            raise ValueError(f"Reduction {reduction} not recognized")

    def f1(self, reduction: str = 'mean') -> torch.Tensor:
        if reduction == 'mean':
            raw_f1_tensor = 2 * self.precision() * self.recall() / (self.precision() + self.recall() + self._epsilon)
            return raw_f1_tensor.mean()
        elif reduction == 'sum':
            raw_f1_tensor = 2 * self.precision() * self.recall() / (self.precision() + self.recall() + self._epsilon)
            return raw_f1_tensor.sum()
        elif reduction == 'none' or reduction is None:
            raw_f1_tensor = 2 * self.precision() * self.recall() / (self.precision() + self.recall() + self._epsilon)
            return raw_f1_tensor
        else:
            raise ValueError(f"Reduction {reduction} not recognized")

    def __repr__(self):
        return f"Meter(TP={self.TP}, TN={self.TN}, FP={self.FP}, FN={self.FN}, total={self.total})"


class BinaryMeter(MulticlassMeter):
    def __init__(self):
        super().__init__(num_classes=2)

    def calculate_metrics(self, reduction='mean', clear=True) -> T.Dict[str, float]:
        return super().calculate_metrics(reduction=reduction, clear=clear, as_scalar=True)
