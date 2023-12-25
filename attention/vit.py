import typing as T
import torch
import torch.nn as nn
import numpy as np
from attention.head import MultiHeadAttention
from attention.patch import PatchGenerator
import copy


class Embedder(nn.Module):
    """
    Embedder module. Generates the linear projection/embedding for the image patches, as well
    as the class embedding (scalar -> vector) and the positional embeddings.

    In a batched learning scenario, it will generate a 3D-tensor of shape (batch size, number of tokens, embedding dimension). In this case, number of tokens is number of patches + 1 (one accounting for the class embedding). Each slice accross dimension 0 of this tensor is the sequence of embeddings corresponding to one full image.
    """

    def __init__(
        self,
        input_h: int,
        input_w: int,
        patch_h: int,
        patch_w: int,
        encoding_dim: int,
        num_channels: int = 3,
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.patcher = PatchGenerator(
            input_h, input_w, patch_h, patch_h, as_sequence=True, flatten=True
        )
        self.N = self.patcher.num_patches
        # patch (C * P * P,) -> embedding (D,)
        self.E = nn.Linear(num_channels * patch_h * patch_w, encoding_dim, bias=False)
        # class embedding
        self.Eclass = nn.Linear(1, encoding_dim)  # TODO should this have a bias?
        # positional embedding (1D only)
        self.Epos = nn.Linear(
            self.N + 1, (self.N + 1) * encoding_dim
        )  # TODO should this have a bias?

    @property
    def total_parameters(self) -> int:
        def get_params(tensor): return np.prod(list(tensor.shape), dtype=int)
        return (
            get_params(self.E.weight)
            + get_params(self.Eclass.weight)
            + get_params(self.Eclass.bias)
            + get_params(self.Epos.weight)
            + get_params(self.Epos.bias)
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if len(label.shape) > 0:  # batched training
            x_class = self.Eclass(torch.unsqueeze(label, dim=-1))  # (B, 1) x (1, D) -> (B, D)
            x_patches = self.patcher.patch(x).to(x.device)  # (B, N, C * h_p * w_p) on same device as image
            batch_size = label.shape[0]
        else:  # unbatched training
            x_class = self.Eclass(torch.unsqueeze(label, dim=0)).unsqueeze(dim=0)  # (1, D)
            x_patches = self.patcher.patch(x).to(x.device).unsqueeze(dim=0)  # (1, N, C * h_p * w_p) on same device as image
            batch_size = 1

        x_class = torch.unsqueeze(x_class, dim=-2)  # (B, 1, D), ready for concat now
        raw_positional_info = torch.arange(0, self.N + 1, dtype=torch.float32, device=x.device)  # (N + 1,)
        x_pos = self.Epos(raw_positional_info).repeat(batch_size, 1)  # (B, N + 1 * D) on same device as image
        x_pos = x_pos.reshape(batch_size, self.N + 1, self.encoding_dim)  # (B, N + 1, D)
        # this patch function will deal just fine with batched images, see implementation
        patch_embeddings = self.E(x_patches)  # (B, N, D)

        batched_final_embeddings = torch.concat([x_class, patch_embeddings], dim=-2) + x_pos  # (B, (N + 1), D)
        return batched_final_embeddings


class ViTBlock(nn.Module):
    """
    ViT block, composed of:
        Layernorm
        Multiheaded self-attention module (where encoding dimension is always the same as input dimension, for the residual connections to work)
        Layernorm
        MLP with GELU non-linearity (as default, but non-linearity can be changed)
    """

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        query_dim: int,
        head_dim: int,
        mlp_layers: T.List[int],
        non_linearity: nn.Module = nn.GELU(),  # must be a module
        init_policy=None,
    ):
        super().__init__()
        if len(mlp_layers) == 0:
            raise ValueError("No linear/fully-connected layer on ViT block")
        elif mlp_layers[-1] != input_dim:
            raise ValueError(
                "Output dimension of MLP does not match output dimension of attention block. This is not possible, because of the existence of a residual connection between the end of the attention block and the end of the MLP."
            )
        self.multihead_att = MultiHeadAttention(
            num_heads=num_heads,
            input_dim=input_dim,
            query_dim=query_dim,
            head_dim=head_dim,
            encoding_dim=input_dim,
            init_policy=init_policy,
        )
        current_dim = input_dim
        self.mlp = nn.ModuleList()
        for layer_size in mlp_layers:
            linear_layer = nn.Linear(current_dim, layer_size)
            if init_policy is not None:
                init_policy(linear_layer.weight)
                init_policy(linear_layer.bias)
            else:
                torch.nn.init.normal_(linear_layer.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(linear_layer.bias, mean=0.0, std=0.02)
            self.mlp.append(linear_layer)
            self.mlp.append(non_linearity)
            current_dim = layer_size
        self.output_dim = current_dim

        self.layernorm1 = nn.LayerNorm(input_dim)  # just before attention block
        self.layernorm2 = nn.LayerNorm(
            input_dim
        )  # just after attention block and before MLP

    @property
    def total_parameters(self) -> int:
        """
        Returns the total number of learnable parameters (weights).
        Assumes the non-linearity is a fixed function and does not have learnable parameters of its own.
        """
        def get_params(tensor):
            return np.prod(list(tensor.shape), dtype=int)
        total = self.multihead_att.total_parameters
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                total += get_params(layer.weight) + get_params(layer.bias)
        total += get_params(self.layernorm1.weight) + get_params(self.layernorm1.bias)
        total += get_params(self.layernorm2.weight) + get_params(self.layernorm2.bias)
        return total

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_prime = z + self.layernorm1(self.multihead_att(z))
        mlp_output = self.layernorm2(z_prime)
        for layer in self.mlp:
            mlp_output = layer(mlp_output)
        mlp_output = mlp_output + z_prime
        return mlp_output


class ViT(nn.Module):
    """
    Customizable ViT (Visual Transformer) encoder (sequence-to-sequence).

    Structure follows closely paper "An image is worth 16 x 16 words: transformers for image recognition at scale", from https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_shape: T.Tuple[int, int, int],
        patch_size: int,
        num_blocks: int,
        num_heads: int,
        embedding_dim: int,
        query_dim: int,
        head_dim: int,
        mlp_layers: T.List[int],
        non_linearity: nn.Module = nn.GELU(),  # must be a module
        init_policy=None,
    ):
        super().__init__()
        if len(img_shape) != 3:
            raise ValueError(
                "Image shape must be the shape of a channel-first 2D image - on the (C, H, W) format"
            )
        self.square_patch_embedder = Embedder(
            *img_shape[1:],
            patch_size,
            patch_size,
            embedding_dim,
            num_channels=img_shape[0]
        )
        self.transformer_blocks = nn.ModuleList(
            [
                ViTBlock(
                    num_heads=num_heads,
                    input_dim=embedding_dim,
                    query_dim=query_dim,
                    head_dim=head_dim,
                    mlp_layers=mlp_layers,
                    non_linearity=non_linearity,
                    init_policy=init_policy,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layernorm = nn.LayerNorm(embedding_dim)

    @property
    def total_parameters(self) -> int:
        def get_params(tensor):
            return np.prod(list(tensor.shape), dtype=int)
        return (
            self.square_patch_embedder.total_parameters
            + sum([t.total_parameters for t in self.transformer_blocks])
            + get_params(self.layernorm.weight)
            + get_params(self.layernorm.bias)
        )

    def forward(self, x: torch.Tensor, label: T.Union[int, float]) -> torch.Tensor:
        """
        Implements ViT algorithm:

        x, label are mapped to a sum of class+patch embeddings [x_class, x_p1,..., x_pN] and positional embeddings x_pos
            z0 = [x_class, x_p1, ..., x_pN] + x_pos ---> of shape ((N + 1) X D)

        then
        for i=1...l
            z_i = ViTBlock_i(z_{i - 1}) ---> of shape ((N + 1) X D), since each ViT block is shape preserving
        and the final result is y = layernorm(z_l^0), an R^D vector obtained taking the first row of the final sequence and normalizing it
        """
        z0 = self.square_patch_embedder(x, label)
        for t in self.transformer_blocks:
            z0 = t(z0)
        return self.layernorm(torch.select(z0, dim=-2, index=0))  # (..., N, D) -> (..., D), taking first D-vector
