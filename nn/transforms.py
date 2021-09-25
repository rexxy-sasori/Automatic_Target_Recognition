"""
Author: Rex Geng

class definitions for customized nn transforms
"""

import torch
from torchvision import transforms as tfd

from nn import quantization


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return data

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        if data.ndim == 3:
            return torch.tensor(data).type(torch.FloatTensor)
        else:
            x = torch.tensor(data)
            x = x.unsqueeze(0)
            return x.type(torch.FloatTensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class QuantInput(object):
    def __init__(self, clip_val, num_bits):
        self.num_bits = num_bits
        self.clip_val = clip_val

    def __call__(self, data):
        data = quantization.quantize_uniform(data, n_bits=self.num_bits, clip=self.clip_val, device='cpu')
        return data


__TRANSFORMS__ = {
    'I': Identity,
    'img2tensor': ToTensor,
    'quant_input': QuantInput,
    'torch_img2tensor': tfd.ToTensor,
    'torch_random_hflip': tfd.RandomHorizontalFlip,
    'torch_random_vflip': tfd.RandomVerticalFlip,
    'torch_random_rotate': tfd.RandomRotation,
    'normalize_mean_std': tfd.Normalize,
}
