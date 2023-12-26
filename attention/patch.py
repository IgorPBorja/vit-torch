import numpy as np
import torch
import warnings

class PatchGenerator:
    """
        Module for, given an image, dividing it in equal rectangular patches.
    """

    def __init__(self,
                 input_height: int,
                 input_width: int,
                 patch_height: int,
                 patch_width: int,
                 flatten: bool = True,
                 as_sequence: bool = True):
        """
            @params:
                input_height (H): height of image
                input_width (W): width of image
                patch_height (h_p): height of patch
                patch_width (w_p): width of patch
                flatten (bool): if true, then flatten each patch from (C, h_p, w_p) to (C * h_p * w_p) before returning
                    Default: true
                as_sequence (bool): if true, return a sequence of N = ceil(H / h_p) * ceil(W / w_p) patches. Else, return a matrix of ceil(H / h_p) x ceil(W / w_p) patches
        """
        self.input_width = input_width
        self.input_height = input_height
        self.patch_width = patch_width
        self.patch_height = patch_height
        if (self.input_width % self.patch_width != 0) or (self.input_height % self.patch_height != 0):
            warnings.warn("Patch dimensions do not divide image dimensions evenly. This will ignore data on borders, effectively cropping the image to the top left ((H // h_p) * h_p) X ((W // w_p) * w_p) patch, were (H, W) are the original dimensions and (h_p, w_p) are the patch dimensions.", UserWarning)
        self.h_ratio = self.input_height // self.patch_height
        self.w_ratio = self.input_width // self.patch_width
        self.num_patches = self.h_ratio * self.w_ratio
        self.flatten = flatten
        self.as_sequence = as_sequence

    def patch(self, x: torch.Tensor):
        """
            * Uses Tensor.unfold to generate patches of size C x patch_height x patch_width (or this, but flattened, if flatten option was initialized as true) from image x.
            * This function also preserves dimensions before (channel, H, W) triplet, if any exist.

            tensor.unfold(dim, size, step) returns all slices of the tensor across a given dimension dim, of size "size", offseting by step each time
                x.unfold(i, size, step) = tensor([x[:,:,...,:,0:size,:,...,:], x[:,:,...,:,step:step+size,:,...,:], ...])

                generates 1 + (d_i - size) // step slices: from [0:size] to [((d_i - size) // step) * step, ((d_i - size) // step) * step + size]
                Dimension of x goes from (d1,...,d_n) to (d_1,..., d_{i-1}, 1 + (d_i - size) // step, d_{i+1},...,d_n, size)
                As a special case, if step == size we have (d1,...,d_n) |-> (d_1,...,d_{i-1}, d_i // size, d_{i + 1},..., d_n, size)
        """
        x = x.moveaxis(-3, -1) ## channel first to channel last without changing data order
        ## divides image in horizontal rectangular strips on each channel
        row_windows = x.unfold(-3, self.patch_height, self.patch_height) ## (..., H // h_p, W, C, h_p)
        patches = row_windows.unfold(-3, self.patch_width, self.patch_width) ## (..., H // h_p, W // w_p, C, h_p, w_p)
        if self.as_sequence:
            patches = patches.reshape((*patches.shape[:-5], patches.shape[-5] * patches.shape[-4], *patches.shape[-3:]))
        if self.flatten:
            patches = patches.reshape((*patches.shape[:-3], patches.shape[-3] * patches.shape[-2] * patches.shape[-1]))
        return patches

