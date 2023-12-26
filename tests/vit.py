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
img_shape = (3, 224, 224)
num_classes = 10
num_samples = 100
num_heads = 8
d = 512
p = 16
num_blocks = 3
mlp_layers = [256, 512]
dq = 256
dv = 256

dataloader_random = [
    (torch.randn(img_shape), torch.tensor(np.random.choice(num_classes))) for i in range(num_samples)
]

dataset_real = data.CIFAR10(root="data",
                            train=True,
                            transform=TR.Compose([TR.ToTensor(), TR.Resize((224, 224))]),
                            download=True)
dataloader_real = DataLoader(dataset_real, batch_size=32, shuffle=True)


def test_vit_block_param_count():
    num_tokens = (img_shape[1] // p) * (img_shape[2] // p) + 1
    vit_block = ViTBlock(num_heads, d, dq, dv, mlp_layers)
    stats = summary(vit_block, input_size=(num_tokens, d),
                    col_names=("output_size", "num_params", "params_percent"),
                    depth=10, device='cpu', row_settings=("depth", "var_names"),
                    mode="train")
    #  print(stats)
    #  print(vit_model.total_parameters, stats.total_params)
    assert vit_block.total_parameters == stats.total_params


def test_seq_block_param_count():
    num_tokens = (img_shape[1] // p) * (img_shape[2] // p) + 1

    class BlockSeq(nn.Module):
        def __init__(self):
            super().__init__()
            #  self.seq = nn.ModuleList()
            #  for i in range(5):
            #      self.seq.append(ViTBlock(num_heads, d, dq, dv, mlp_layers))
            self.seq = nn.ModuleList([ViTBlock(num_heads, d, dq, dv, mlp_layers) for i in range(5)])

        def forward(self, z):
            for block in self.seq:
                z = block(z)
            return z

        @property
        def total_parameters(self):
            return sum([block.total_parameters for block in self.seq])

    seq = BlockSeq()
    stats = summary(seq, input_size=(num_tokens, d),
                    col_names=("output_size", "num_params", "params_percent"),
                    depth=10, device='cpu', row_settings=("depth", "var_names"),
                    mode="train")
    #  print(stats)
    #  print(vit_model.total_parameters, stats.total_params)
    assert seq.total_parameters == stats.total_params


def test_vit_param_count():
    vit_model = ViT(img_shape, p, num_blocks, num_heads, d, dq, dv, mlp_layers)
    # depth is the depth for recursive introspection of modules
    #  print(vit_model)
    stats = summary(vit_model, input_size=[(1, 3, 224, 224), (1,)],
                    col_names=("output_size", "num_params", "params_percent"),
                    depth=10, device='cpu', row_settings=("depth", "var_names"),
                    mode="train")
    #  print(stats)
    #  print(vit_model.total_parameters, stats.total_params)

    get_params = lambda tensor: np.prod(list(tensor.shape), dtype=int)
    print(f"Embedder: {vit_model.square_patch_embedder.total_parameters} params")
    for i, t in enumerate(vit_model.transformer_blocks):
        print(f"Block {i}: {t.total_parameters} params")
    print(f"Final layernorm: {get_params(vit_model.layernorm.weight) + get_params(vit_model.layernorm.bias)} params")

    assert vit_model.total_parameters == stats.total_params


@pytest.mark.parametrize("dataset", [dataloader_random, dataloader_real])
def test_vit_shape(dataset):
    vit_model = ViT(img_shape, p, num_blocks, num_heads, d, dq, dv, mlp_layers)
    vit_model = vit_model.to("cuda")
    print(f"Total number of learnable parameters are: {vit_model.total_parameters}")
    with torch.no_grad():
        for img, label in tqdm(dataset):
            batch_size = 1 if len(img.shape) == 3 else img.shape[0]
            img = img.to("cuda")
            label = label.to(torch.float32).to("cuda")
            out = vit_model(img, label)
            assert out.shape == (batch_size, d)


def run_model(dataset, num_epochs: int) -> tuple[float, list[float]]:
    vit_encoder = ViT(img_shape, p, num_blocks, num_heads, d, dq, dv, mlp_layers)
    vit_encoder = vit_encoder.to("cuda")
    decoder = nn.Sequential(nn.Linear(d, num_classes), nn.Softmax(dim=-1))
    decoder = decoder.to("cuda")

    complete_pipeline = nn.Sequential(vit_encoder, decoder)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    adam = optim.Adam(complete_pipeline.parameters(), lr=0.01)

    # in some small number of epochs, it must reduce loss
    initial_loss = 0.0
    with torch.no_grad():
        for img, label in tqdm(dataset):
            img = img.to('cuda')
            float_label = label.to(torch.float32).to("cuda")
            one_hot_label = nn.functional.one_hot(torch.tensor(label), num_classes=num_classes).to(torch.float32).to('cuda')
            out = vit_encoder(img, float_label)
            pred = decoder(out)
            initial_loss += criterion(one_hot_label, pred).item()
    print(f"Initial loss (untrained model): {initial_loss}")

    losses = []
    running_loss = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for img, label in tqdm(dataset):
            img = img.to('cuda')
            float_label = label.to(torch.float32).to("cuda")
            out = vit_encoder(img, float_label)
            pred = decoder(out)
            one_hot_label = nn.functional.one_hot(torch.tensor(label), num_classes=num_classes).to(torch.float32).to('cuda')
            loss = criterion(one_hot_label, pred)

            loss.backward()
            adam.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}: loss is {running_loss}")
        losses.append(running_loss)
    return initial_loss, losses


@pytest.mark.parametrize("dataset", [dataloader_random, dataloader_real])
def test_vit_loss_very_weak(dataset):
    """
        Assert if loss changed at all
    """
    initial_loss, losses = run_model(dataset, NUM_WEAK_TEST_EPOCHS)

    assert losses[-1] != initial_loss


@pytest.mark.parametrize("dataset", [dataloader_random, dataloader_real])
def test_vit_loss_weak(dataset):
    """
        Assert if loss decreased at all
    """
    initial_loss, losses = run_model(dataset, NUM_WEAK_TEST_EPOCHS)

    assert losses[-1] < initial_loss


@pytest.mark.parametrize("dataset,factor", itertools.product([dataloader_random, dataloader_real], [0.1, 0.2, 0.3]))
def test_vit_loss_stronger(dataset, factor: float):
    """
        Assert if loss decreased by a factor of more than <factor>
    """
    initial_loss, losses = run_model(dataset, NUM_STRONG_TEST_EPOCHS)
    assert losses[-1] < (1.0 - factor) * initial_loss
