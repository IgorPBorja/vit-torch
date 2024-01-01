import typing as T
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision
from torchvision import transforms as TR
from torchvision.datasets import CIFAR10
import numpy as np
from attention.patch import PatchGenerator
from attention.head import AttentionHead, MultiHeadAttention
from attention.vit import ViT, ViTBlock
from benchmark.meters import MulticlassMeter
from tqdm import tqdm


def train_step(encoder: nn.Module,
               decoder: nn.Module,
               dataloader: DataLoader,
               criterion,
               opt: torch.optim.Adam,
               meter: MulticlassMeter,
               *,
               device: T.Union[torch.device, str]  = "cpu") -> T.Tuple[float, T.Dict[str, float]]:
    encoder.train()
    decoder.train()
    running_loss = 0.0
    opt.zero_grad()
    for (imgs, labels) in tqdm(dataloader, total=len(dataloader)):
        imgs = imgs.to(device)
        float_labels = labels.to(torch.float32).to(device)
        embeddings = encoder(imgs, float_labels)
        logits = decoder(embeddings)
        one_hot_labels = nn.functional.one_hot(labels, num_classes=10).to(torch.float32).to(device)

        loss = criterion(logits, one_hot_labels)
        loss.backward()
        opt.step()

        meter.update(torch.softmax(logits, dim=-1).cpu(), labels, as_probabilities=True)
        running_loss += loss.cpu().item() * imgs.shape[0]
    return running_loss, meter.calculate_metrics(reduction='mean', clear=True, as_scalar=True)


def eval_step(encoder: nn.Module,
              decoder: nn.Module,
              dataloader: DataLoader,
              criterion,
              meter: MulticlassMeter,
              *,
              device: T.Union[torch.device, str] = "cpu") -> T.Tuple[float, T.Dict[str, float]]:
    # assumes both encoder and decoder already allocated in correct device
    encoder.eval()
    decoder.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (imgs, labels) in tqdm(dataloader, total=len(dataloader)):
            imgs = imgs.to(device)
            float_labels = labels.to(torch.float32).to(device)
            embeddings = encoder(imgs, float_labels)
            logits = decoder(embeddings)
            one_hot_labels = nn.functional.one_hot(labels, num_classes=10).to(torch.float32).to(device)
            loss = criterion(logits, one_hot_labels)

            meter.update(torch.softmax(logits, dim=-1).cpu(), labels, as_probabilities=True)
            running_loss += loss.cpu().item() * imgs.shape[0]
    return running_loss, meter.calculate_metrics(reduction='mean', clear=True, as_scalar=True)


def trainloop(train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              encoder: nn.Module,
              decoder: nn.Module,
              criterion,
              opt: torch.optim.Adam,
              num_epochs: int,
              num_classes: int,
              scheduler=None,
              *,
              verbosity: int = 1,
              device: T.Union[torch.device, str] = torch.device("cpu"),
              reduction: str = "mean") -> T.Tuple[list[float], list[float], list[T.Dict[str, float]], list[T.Dict[str, float]]]:
    """
        Train a model for a specified number of epochs, optionally with a learning rate scheduler.

        :param train_dataloader [torch.utils.data.dataloader.DataLoader]: dataloader for training data
        :param val_dataloader [torch.utils.data.dataloader.DataLoader]: dataloader for validation data
        :param encoder [torch.nn.Module]: encoder / feature extractor
        :param decoder [torch.nn.Module]: decoder / classifier
        :param criterion: loss function
        :param opt [torch.optim.Optimizer]: optimizer
        :param num_epochs [int]: number of epochs to train for
        :param num_classes [int]: number of classes in dataset
        :param scheduler [torch.optim.lr_scheduler._LRScheduler]: learning rate scheduler
        :keyword verbosity [int]: verbosity level. 0 = no output, 1 = print metrics, >=2 = print metrics and loss
        :keyword device [torch.device]: device to run training on. Defaults to cpu
    """
    train_losses = []
    val_losses = []
    train_history = []
    val_history = []
    meter = MulticlassMeter(num_classes=num_classes)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = train_step(encoder, decoder, train_dataloader, criterion, opt, meter, device=device)
        if verbosity > 1:
            print(f"Epoch {epoch}: train loss = {train_loss}")
        if verbosity > 0:
            for metric, val in train_metrics.items():
                print(f"Epoch {epoch}: train {metric} = {val}")
            print()
        val_loss, val_metrics = eval_step(encoder, decoder, val_dataloader, criterion, meter, device=device)
        if verbosity > 1:
            print(f"Epoch {epoch}: val loss = {val_loss}")
        if verbosity > 0:
            for metric, val in val_metrics.items():
                print(f"Epoch {epoch}: val {metric} = {val}")
            print()
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        print(f"Epoch {epoch}: train loss is {train_loss}, val loss is {val_loss}")
        if reduction == 'mean':
            train_losses.append(train_loss / len(train_dataloader.dataset))
            val_losses.append(val_loss / len(val_dataloader.dataset))
        elif reduction == 'sum':
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        else:
            raise ValueError(f"Reduction '{reduction}' not recognized")
        if scheduler is not None:
            scheduler.step()  # FIXME make scheduler step on each step instead of each epoch
    return train_losses, val_losses, train_history, val_history
