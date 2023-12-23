import typing as T
import numpy as np
import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    """
        Single-headed attention module
    """
    def __init__(self, 
                 input_dim: int, 
                 query_dim: int, 
                 encoding_dim: T.Optional[int] = None,
                 init_policy = None):
        """
            @params:
                input_dim (d): dimension of input sequence
                query_dim (d_k): dimension of query and key vectors
                encoding_dim (d_v): final dimension of output sequence of attention block.
                    Default (if no value provided): equal to input_dim
                init_policy: function to initialize modules. If None, defaults to random initialization with mean=0 and std=0.02
                    Recommended: use functions from module torch.nn.init
        """
        super().__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        if encoding_dim is not None:
            self.encoding_dim = encoding_dim
        else:
            self.encoding_dim = self.input_dim

        self.query = nn.Linear(self.input_dim, self.query_dim, bias=False)
        self.key = nn.Linear(self.input_dim, self.query_dim, bias=False)
        self.value = nn.Linear(self.input_dim, self.encoding_dim, bias=False)
        if init_policy is not None:
            with torch.no_grad():
                init_policy(self.query.weight)
                init_policy(self.key.weight)
                init_policy(self.value.weight)
            ## raise NotImplementedError("Custom iniatialization policy not yet implemented")
        else:
            ## by default all functions on nn.init execute under no_grad
            ## https://pytorch.org/docs/stable/nn.init.html
            nn.init.normal_(self.query.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.key.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.value.weight, mean=0.0, std=0.02)

    def forward(self, z : torch.Tensor):
        """
            Implements scaled attention algorithm

            output = softmax(qk^T / sqrt(d_query)) * v
            where q = z * U_q, k = z * U_k and v = z * U_v
        """
        ## TODO analise possibility of batched learning in this forward
        assert(len(z.shape) == 2) ## N x d_in
        q = self.query.forward(z) ## N x d_q
        k = self.key.forward(z) ## N x d_q
        v = self.value.forward(z) ## N x d_v
        scaled_query = torch.softmax(q @ k.transpose(-2, -1), dim=-1) * (self.query_dim) ** (- 0.5)
        return scaled_query @ v

    @property
    def total_parameters(self) -> int:
        """
            Returns total number of parameters (weights from q, k and v matrices)
        """
        total = 0
        for tensor in [self.query.weight, self.key.weight, self.value.weight]:
            total += np.prod(list(tensor.shape), dtype=int)

        return total
