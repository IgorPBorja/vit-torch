import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision.datasets as data
import torchvision.transforms as TR
from torch.utils.data.dataloader import DataLoader
from attention.vit import ViT, ViTBlock
import numpy as np
from tqdm import tqdm
import pytest
import itertools
from torchinfo import summary

NUM_WEAK_TEST_EPOCHS = 1
NUM_STRONG_TEST_EPOCHS = 10
STRONG_FACTOR = 0.5
NUM_RANDOM_SAMPLES = 100
img_shape = (3, 32, 32)
num_classes = 10
num_heads = 8
d = 512
p = 8
num_blocks = 3
mlp_layers = [256, 512]
dq = 256
dv = 256

dataloader_random = [
    (torch.randn(img_shape), torch.tensor(np.random.choice(num_classes))) for i in range(NUM_RANDOM_SAMPLES)
]

dataset_real = data.CIFAR10(root="data",
                            train=True,
                            transform=TR.Compose([TR.ToTensor(), TR.Resize(img_shape[1:]), TR.Normalize((0.5,), (0.5,))]),
                            download=True)
dataloader_real = DataLoader(dataset_real, batch_size=64, shuffle=False)  # shuffle=False so reproducible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_vit_block_param_count():
    num_tokens = (img_shape[1] // p) * (img_shape[2] // p) + 1
    vit_block = ViTBlock(num_heads, d, dq, dv, mlp_layers)
    stats = summary(vit_block, input_size=(num_tokens, d),
                    col_names=("output_size", "num_params", "params_percent"),
                    depth=10, device='cpu', row_settings=("depth", "var_names"),
                    mode="train")
    assert vit_block.total_parameters == stats.total_params


def test_vit_param_count():
    vit_model = ViT(img_shape, p, num_blocks, num_heads, d, dq, dv, mlp_layers)
    # depth is the depth for recursive introspection of modules
    stats = summary(vit_model, input_size=[(1, *img_shape), (1,)],
                    col_names=("output_size", "num_params", "params_percent"),
                    depth=10, device='cpu', row_settings=("depth", "var_names"),
                    mode="train")

    def get_params(tensor): return np.prod(list(tensor.shape), dtype=int)
    print(f"Embedder: {vit_model.square_patch_embedder.total_parameters} params")
    for i, t in enumerate(vit_model.transformer_blocks):
        print(f"Block {i}: {t.total_parameters} params")
    print(f"Final layernorm: {get_params(vit_model.layernorm.weight) + get_params(vit_model.layernorm.bias)} params")

    assert vit_model.total_parameters == stats.total_params


@pytest.mark.parametrize("dataset", [dataloader_random, dataloader_real])
def test_vit_shape(dataset):
    vit_model = ViT(img_shape, p, num_blocks, num_heads, d, dq, dv, mlp_layers)
    vit_model = vit_model.to(device)
    print(f"\nTotal number of learnable parameters are: {vit_model.total_parameters}")
    with torch.no_grad():
        img, label = next(iter(dataset))
        batch_size = 1 if len(img.shape) == 3 else img.shape[0]
        img = img.to(device)
        label = label.to(torch.float32).to(device)
        out = vit_model(img, label)
        assert out.shape == (batch_size, d)


def train_epoch(dataloader,
                vit_encoder: ViT,
                decoder: nn.Module,
                criterion: nn.Module,
                do_backward_step: bool = False,
                optimizer=None) -> float:
    num_imgs = sum([len(batch) if len(batch.shape) > 3 else 1 for batch, _ in dataloader])
    running_loss: float = 0.0
    for i, (img, label) in enumerate(tqdm(dataloader, total=len(dataloader))):
        if optimizer is not None:
            optimizer.zero_grad()
        img = img.to(device)
        float_label = label.to(torch.float32).to(device)
        one_hot_label = nn.functional.one_hot(label, num_classes=num_classes).to(torch.float32).to(device)
        if len(one_hot_label.shape) == 1:
            one_hot_label = one_hot_label.unsqueeze(dim=0)  # add batch dimension (unbatched learning)
        out = vit_encoder(img, float_label)
        pred = decoder(out)
        loss = criterion(pred, one_hot_label)
        if do_backward_step:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
    return running_loss / num_imgs


def run_model(dataset, num_epochs: int) -> tuple[float, list[float]]:
    vit_encoder = ViT(img_shape, p, num_blocks, num_heads, d, dq, dv, mlp_layers, nn.LeakyReLU, non_linearity_params={'negative_slope': 0.02})
    vit_encoder = vit_encoder.to(device)
    decoder = nn.Linear(d, num_classes)
    decoder = decoder.to(device)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    adam = optim.Adam(list(vit_encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    # in some small number of epochs, it must reduce loss
    print("\n" + 80 * "=")
    print(f"Total (learnable) parameters: {vit_encoder.total_parameters + np.prod(list(decoder.weight.shape)) + np.prod(list(decoder.bias.shape))}")
    with torch.no_grad():
        initial_loss = train_epoch(dataset, vit_encoder, decoder, criterion)
    print(f"Initial loss (untrained model): {initial_loss}")

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(dataset, vit_encoder, decoder, criterion, do_backward_step=True, optimizer=adam)
        print(f"Epoch {epoch}: loss is {epoch_loss}")
        losses.append(epoch_loss)
    print(80 * "=")
    return initial_loss, losses


@pytest.mark.parametrize("dataset", [dataloader_random, dataloader_real])
def test_vit_loss_weak(dataset):
    """
        Assert if loss decreased at all
    """
    initial_loss, losses = run_model(dataset, NUM_WEAK_TEST_EPOCHS)

    assert losses[-1] < initial_loss


@pytest.mark.parametrize("dataset,factor", itertools.product([dataloader_random, dataloader_real], [STRONG_FACTOR]))
def test_vit_loss_stronger(dataset, factor: float):
    """
        Assert if loss decreased by a factor of more than <factor>
    """
    initial_loss, losses = run_model(dataset, NUM_STRONG_TEST_EPOCHS)
    assert losses[-1] < (1.0 - factor) * initial_loss
