from benchmark.pipeline import trainloop
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as TR
from torchvision.datasets import CIFAR10
from attention.vit import ViT

BATCH_SIZE = 64
IMG_SHAPE = (3, 32, 32)
NUM_BLOCKS = 3
NUM_HEADS = 8
D = 512
DQ = 256
DV = 256
MLP_LAYERS = [256, 512]
PATCH_SIZE = 8
NUM_CLASSES = 10
NUM_EPOCHS = 10
LR = 1e-5


def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CIFAR10(root="./data", train=True, download=True,
                            transform=TR.Compose([TR.ToTensor(), TR.Resize(IMG_SHAPE[1:]), TR.Normalize((0.5,), (0.5,))]))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = CIFAR10(root="./data", train=False, download=True,
                           transform=TR.Compose([TR.ToTensor(), TR.Resize(IMG_SHAPE[1:]), TR.Normalize((0.5,), (0.5,))]))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    vit_encoder = ViT(IMG_SHAPE, patch_size=PATCH_SIZE, num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS, embedding_dim=D, query_dim=DQ, head_dim=DV, mlp_layers=MLP_LAYERS)
    decoder = nn.Linear(D, NUM_CLASSES)
    print(f"Model total parameters: {sum([p.numel() for p in vit_encoder.parameters()]) + sum([p.numel() for p in decoder.parameters()])}")

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(vit_encoder.parameters()) + list(decoder.parameters()), lr=LR)
    train_losses, val_losses, train_history, val_history = trainloop(train_dataloader, test_dataloader, vit_encoder, decoder, criterion, opt, NUM_EPOCHS, num_classes=10, verbosity=float('inf'), device=device)


benchmark()
