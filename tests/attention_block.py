import typing as T
import torch
from torch import nn
import torchvision
import torchvision.transforms as TR
import numpy as np
from attention.head import AttentionHead
import pytest
from tqdm import tqdm
from attention.patch import PatchGenerator

NUM_SHAPE_TESTS = 5
NUM_CONV_TESTS = 3

TEST_DATA_SHAPE = [
    (np.random.randint(1, 100),
     np.random.randint(1, 1000),
     np.random.randint(1, 1000),
     np.random.randint(1, 1000),
     np.random.randint(1, 100))
    for i in range(NUM_SHAPE_TESTS)]

TEST_DATA_CONV_ATT = [
    ((32, 32),
     dq,
     dv,
     5,
     0.9) for dq, dv in [(256, 512), (512, 256), (784, 360)]]


def eval_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              criterion,
              preprocess: T.Callable = lambda x: x) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (imgs, labels) in tqdm(dataloader, total=len(dataloader)):
            processed_imgs = preprocess(imgs)
            out = model(processed_imgs)
            one_hot_labels = nn.functional.one_hot(labels, num_classes=10).to(torch.float32)
            loss = criterion(out, one_hot_labels)
            running_loss += loss.item() * imgs.shape[0]
    return running_loss


def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion,
               opt: torch.optim.Adam,
               preprocess: T.Callable = lambda x: x) -> float:
    model.train()
    running_loss = 0.0
    opt.zero_grad()
    for (imgs, labels) in tqdm(dataloader, total=len(dataloader)):
        processed_imgs = preprocess(imgs)
        out = model(processed_imgs)
        one_hot_labels = nn.functional.one_hot(labels, num_classes=10).to(torch.float32)
        loss = criterion(out, one_hot_labels)
        loss.backward()
        opt.step()
        running_loss += loss.item() * imgs.shape[0]
    return running_loss


def pipeline(model, dataloader, criterion, opt, num_epochs, preprocess=lambda x: x):
    num_images = sum([len(batch) for batch in dataloader])
    initial_loss = eval_step(model, dataloader, criterion, preprocess=preprocess) / num_images
    print(f"Initial per-image loss = {initial_loss}")
    losses = []
    for i in range(num_epochs):
        new_epoch_loss = train_step(model, dataloader, criterion, opt, preprocess=preprocess) / num_images
        losses.append(new_epoch_loss)
        print(f"Epoch {i}: per-image loss = {new_epoch_loss}")
    return initial_loss, losses


@pytest.mark.parametrize("batch_size, input_dim, query_dim, encoding_dim, num_tokens", TEST_DATA_SHAPE)
def test_shape(batch_size, input_dim, query_dim, encoding_dim, num_tokens):
    att = AttentionHead(input_dim=input_dim, query_dim=query_dim, encoding_dim=encoding_dim)
    input_seq = torch.randn((batch_size, num_tokens, input_dim))
    output_seq = att(input_seq)
    assert output_seq.shape == torch.Size((batch_size, num_tokens, encoding_dim))


@pytest.mark.parametrize("input_dim, query_dim, encoding_dim, num_epochs, convergence_factor", TEST_DATA_CONV_ATT)
def test_attention_cnn_cifar10(input_dim, query_dim, encoding_dim, num_epochs, convergence_factor):
    BATCH_SIZE = 64
    PATCH_SIZE = 8
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                           transform=TR.Compose(
                                               [TR.ToTensor(),
                                                TR.Resize(input_dim),
                                                TR.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    patcher = PatchGenerator(*input_dim, PATCH_SIZE, PATCH_SIZE)  # 64 patches --> (batch_size, 64, 3 * 4 * 4)
    num_patches = (input_dim[0] // PATCH_SIZE) * (input_dim[1] // PATCH_SIZE)
    imgs, _ = next(iter(dataloader))
    assert patcher.patch(imgs).shape == (BATCH_SIZE, num_patches, 3 * PATCH_SIZE ** 2)

    att = AttentionHead(input_dim=3 * PATCH_SIZE * PATCH_SIZE, query_dim=query_dim, encoding_dim=encoding_dim)
    decoder = torch.nn.Linear(num_patches * encoding_dim, 10)
    model = nn.Sequential(att, nn.Flatten(), decoder)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    print(80 * "=")
    print(f"input_dim d = {input_dim}, query dimension d_q = {query_dim}, output/encoding dimension d_v = {encoding_dim}")
    print(f"Target loss reduction: {100 * (1 - convergence_factor):.2f}%")
    print(f"Total parameters = {att.total_parameters + np.prod(decoder.weight.shape, dtype=int) + np.prod(decoder.bias.shape, dtype=int)}")
    print(f"\tAttention parameters = {att.total_parameters}")
    print(f"\tDecoder parameters = {np.prod(decoder.weight.shape, dtype=int) + np.prod(decoder.bias.shape, dtype=int)}")
    initial_loss, losses = pipeline(model, dataloader, criterion, opt, num_epochs, preprocess=patcher.patch)
    print(80 * "=")
    assert losses[-1] < convergence_factor * initial_loss
