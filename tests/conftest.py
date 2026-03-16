import sys
import types


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
torch.hann_window = lambda n_fft, device=None, dtype=None: DummyTensor((n_fft,), device, dtype)
torch.linspace = lambda start, end, steps, device=None, dtype=None: DummyTensor(  # noqa: E731
    (steps,), device, dtype
)
torch.stft = lambda waveform, **kwargs: DummyTensor(  # noqa: E731
    (
        waveform.shape[0],
        kwargs['n_fft'] // 2 + 1,
        max(2, waveform.shape[-1] // kwargs['hop_length']),
    ),
    device=waveform.device,
    dtype=waveform.dtype,
)
torch.istft = lambda spec, **kwargs: DummyTensor(  # noqa: E731
    (spec.shape[0], kwargs['length']),
    device=spec.device,
    dtype='float32',
)

torchaudio = types.ModuleType('torchaudio')
torchaudio.save = lambda buffer, audio_tensor, sample_rate, format: buffer.write(  # noqa: E731
    f'{format}:{sample_rate}:{audio_tensor.shape[-1]}'.encode()
)
torchaudio.functional = types.SimpleNamespace(
    phase_vocoder=lambda spectrogram, rate, phase_advance: DummyTensor(  # noqa: E731
        (
            spectrogram.shape[0],
            spectrogram.shape[1],
            max(1, int(round(spectrogram.shape[2] / rate))),
        ),
        device=spectrogram.device,
        dtype=spectrogram.dtype,
    )
)

sys.modules.setdefault('torch', torch)
sys.modules.setdefault('torchaudio', torchaudio)
