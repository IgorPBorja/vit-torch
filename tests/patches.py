from attention.patch import PatchGenerator
import torch
import numpy as np
import pytest

even_testdata = [
        (224, 224, 8, 14),
        (224, 224, 28, 28),
        (224, 224, 1, 1)
        ]

uneven_testdata = [
        (224, 224, 75, 33),
        (224, 224, 3, 7)
        ]

@pytest.mark.parametrize("h,w,hp,wp", even_testdata + uneven_testdata)
def test_shape(h, w, hp, wp):
    a = torch.randn((3, h, w))
    hratio, wratio = h // hp, w // wp

    gen = PatchGenerator(h, w, hp, wp, as_sequence=False, flatten=False)
    assert(gen.patch(a).shape == torch.Size((hratio, wratio, 3, hp, wp)))

    gen = PatchGenerator(h, w, hp, wp, as_sequence=True, flatten=False)
    assert(gen.patch(a).shape == torch.Size((hratio * wratio, 3, hp, wp)))

    gen = PatchGenerator(h, w, hp, wp, as_sequence=False, flatten=True)
    assert(gen.patch(a).shape == torch.Size((hratio, wratio, 3 * hp * wp)))

    gen = PatchGenerator(h, w, hp, wp, as_sequence=True, flatten=True)
    assert(gen.patch(a).shape == torch.Size((hratio * wratio, 3 * hp * wp)))

@pytest.mark.parametrize("h,w,hp,wp", even_testdata)
def test_content(h, w, hp, wp):
    """
        Assert that the patches correctly correspond with image data.
    """
    def check_patch(a, b, patch_i, patch_j, hp, wp, as_sequence: bool = True, flatten: bool = True):
        hratio = a.shape[1] // hp
        wratio = a.shape[2] // wp
        for c in range(a.shape[0]):
            for i in range(hp):
                for j in range(wp):
                    patch_index = (patch_i, patch_j) if not as_sequence else (patch_i * wratio + patch_j,)
                    inner_index = (c, i, j) if not flatten else (c * hp * wp + i * wp + j,)
                    final_index = (*patch_index, *inner_index)
                    row_index, col_index = patch_i * hp + i, patch_j * wp + j
                    assert a[c,row_index,col_index] == b[final_index]
                    
    for as_sequence in (False, True):
        for flatten in (False, True):
            a = torch.randn((3, h, w)) ## (3, h, w)
            gen = PatchGenerator(h, w, hp, wp, as_sequence=as_sequence, flatten=flatten)
            b = gen.patch(a)
            for i in range(h // hp):
                for j in range(w // wp):
                    check_patch(a, b, i, j, hp, wp, as_sequence, flatten)
