import sys
import types

import numpy as np


class DummyTensor:
    def __init__(self, shape, device='cpu', dtype='float32'):
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        self.is_cuda = device == 'cuda'

    def dim(self):
        return len(self.shape)

    def unsqueeze(self, dim):
        dim = len(self.shape) + dim + 1 if dim < 0 else dim
        shape = list(self.shape)
        shape.insert(dim, 1)
        return DummyTensor(shape, device=self.device, dtype=self.dtype)

    def squeeze(self, dim=None):
        shape = list(self.shape)
        if dim is None:
            shape = [size for size in shape if size != 1] or [1]
        elif shape[dim] == 1:
            shape.pop(dim)
        return DummyTensor(shape, device=self.device, dtype=self.dtype)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def cpu(self):
        return DummyTensor(self.shape, device='cpu', dtype=self.dtype)

    def numpy(self):
        return np.zeros(self.shape, dtype=np.float32)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        shape = list(self.shape)
        new_shape = []
        dim_idx = 0
        for item in key:
            if item is None:
                new_shape.append(1)
                continue
            if isinstance(item, slice):
                start, stop, step = item.indices(shape[dim_idx])
                size = max(0, (stop - start + (step - 1)) // step)
                new_shape.append(size)
                dim_idx += 1
                continue
            dim_idx += 1

        new_shape.extend(shape[dim_idx:])
        return DummyTensor(new_shape, device=self.device, dtype=self.dtype)


torch = types.ModuleType('torch')
torch.Tensor = DummyTensor
torch.int16 = 'int16'


def _from_numpy(arr):
    return DummyTensor(arr.shape, device='cpu', dtype='float32')


torch.from_numpy = _from_numpy

torchaudio = types.ModuleType('torchaudio')
torchaudio.save = lambda buffer, audio_tensor, sample_rate, format: buffer.write(  # noqa: E731
    f'{format}:{sample_rate}:{audio_tensor.shape[-1]}'.encode()
)
torchaudio.functional = types.SimpleNamespace()

sys.modules.setdefault('torch', torch)
sys.modules.setdefault('torchaudio', torchaudio)
