import numpy as np
from flask import Response

from app import create_app
from app.services import audio


class DummyTTS:
    sample_rate = 24000

    def validate_voice(self, voice):
        return True, f'voice ok: {voice}'

    def list_voices(self):
        return [{'id': 'alba'}]

    def get_voice_state(self, voice):
        return {'voice': voice}

    def generate_audio(self, voice_state, text):
        return audio.torch.Tensor((1, 48000))

    def generate_audio_stream(self, voice_state, text):
        yield audio.torch.Tensor((1, 24000))


def test_generate_speech_rejects_non_numeric_speed(monkeypatch):
    import app.routes as routes

    monkeypatch.setattr(routes, 'get_tts_service', lambda: DummyTTS())
    client = create_app({'TESTING': True}).test_client()

    response = client.post('/v1/audio/speech', json={'input': 'hello', 'speed': 'fast'})

    assert response.status_code == 400
    assert response.get_json() == {'error': "'speed' must be a number"}


def test_generate_speech_rejects_out_of_range_speed(monkeypatch):
    import app.routes as routes

    monkeypatch.setattr(routes, 'get_tts_service', lambda: DummyTTS())
    client = create_app({'TESTING': True}).test_client()

    response = client.post('/v1/audio/speech', json={'input': 'hello', 'speed': 4.5})

    assert response.status_code == 400
    assert response.get_json() == {'error': "'speed' must be between 0.25 and 4.0"}


def test_generate_speech_passes_speed_to_file_generation(monkeypatch):
    import app.routes as routes

    captured = {}

    def fake_generate_file(tts, voice_state, text, fmt, speed):
        captured.update(voice_state=voice_state, text=text, fmt=fmt, speed=speed)
        return Response(b'ok', mimetype='audio/mpeg')

    monkeypatch.setattr(routes, 'get_tts_service', lambda: DummyTTS())
    monkeypatch.setattr(routes, '_generate_file', fake_generate_file)
    client = create_app({'TESTING': True, 'STREAM_DEFAULT': False}).test_client()

    response = client.post(
        '/v1/audio/speech',
        json={'input': 'hello', 'voice': 'alba', 'speed': 1.5, 'response_format': 'mp3'},
    )

    assert response.status_code == 200
    assert captured == {
        'voice_state': {'voice': 'alba'},
        'text': 'hello',
        'fmt': 'mp3',
        'speed': 1.5,
    }


def test_generate_speech_passes_speed_to_streaming(monkeypatch):
    import app.routes as routes

    captured = {}

    def fake_stream_audio(tts, voice_state, text, fmt, speed):
        captured.update(voice_state=voice_state, text=text, fmt=fmt, speed=speed)
        return Response(b'ok', mimetype='audio/L16')

    monkeypatch.setattr(routes, 'get_tts_service', lambda: DummyTTS())
    monkeypatch.setattr(routes, '_stream_audio', fake_stream_audio)
    client = create_app({'TESTING': True, 'STREAM_DEFAULT': False}).test_client()

    response = client.post(
        '/v1/audio/speech',
        json={
            'input': 'hello',
            'voice': 'alba',
            'speed': 0.75,
            'response_format': 'pcm',
            'stream': True,
        },
    )

    assert response.status_code == 200
    assert captured == {
        'voice_state': {'voice': 'alba'},
        'text': 'hello',
        'fmt': 'pcm',
        'speed': 0.75,
    }


def test_convert_audio_applies_speed_before_encoding(monkeypatch):
    calls = {}

    def fake_apply_speed(audio_tensor, speed, sample_rate):
        calls['speed'] = speed
        calls['sample_rate'] = sample_rate
        return audio.torch.Tensor((1, 32000))

    monkeypatch.setattr(audio, 'apply_speed', fake_apply_speed)

    buffer = audio.convert_audio(audio.torch.Tensor((1, 48000)), 24000, 'wav', speed=1.5)

    assert calls == {'speed': 1.5, 'sample_rate': 24000}
    assert buffer.read() == b'wav:24000:32000'


def test_apply_speed_uses_wsola():
    """apply_speed should use WSOLA time-stretching, producing shorter output for speed > 1."""
    # Create a short sine wave so the stretcher has real data to work with
    sr = 24000
    duration = 0.2  # 200 ms
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    sine = np.sin(2 * np.pi * 440 * t)

    # Wrap as a real torch-like tensor via numpy → DummyTensor replacement
    import torch as stub_torch

    class RealishTensor:
        """Thin wrapper around a numpy array that satisfies apply_speed's interface."""

        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.float32)
            self.shape = self._arr.shape
            self.device = 'cpu'
            self.dtype = 'float32'
            self.is_cuda = False

        def dim(self):
            return self._arr.ndim

        def unsqueeze(self, d):
            return RealishTensor(np.expand_dims(self._arr, d))

        def squeeze(self, d=None):
            return RealishTensor(np.squeeze(self._arr, axis=d))

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def __getitem__(self, key):
            return RealishTensor(self._arr[key])

        def to(self, **kwargs):
            return self

    # Monkeypatch torch.from_numpy for this test
    orig_from_numpy = stub_torch.from_numpy
    stub_torch.from_numpy = lambda arr: RealishTensor(arr)

    try:
        tensor_in = RealishTensor(sine.reshape(1, -1))
        result = audio.apply_speed(tensor_in, speed=1.5, sample_rate=sr)

        # Output should be roughly n_samples / 1.5 in length (±20 %)
        expected = n_samples / 1.5
        assert result.shape[-1] < n_samples, 'speed > 1 should shorten audio'
        assert abs(result.shape[-1] - expected) / expected < 0.20
    finally:
        stub_torch.from_numpy = orig_from_numpy


def test_stream_speed_chunks_matches_full_apply_speed():
    sr = 24000
    speed = 1.25
    duration = 0.8
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    signal = (0.7 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)).astype(
        np.float32
    )

    import torch as stub_torch

    class RealishTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.float32)
            self.shape = self._arr.shape
            self.device = 'cpu'
            self.dtype = 'float32'
            self.is_cuda = False

        def dim(self):
            return self._arr.ndim

        def unsqueeze(self, d):
            return RealishTensor(np.expand_dims(self._arr, d))

        def squeeze(self, d=None):
            return RealishTensor(np.squeeze(self._arr, axis=d))

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def __getitem__(self, key):
            return RealishTensor(self._arr[key])

        def to(self, **kwargs):
            return self

    orig_from_numpy = stub_torch.from_numpy
    stub_torch.from_numpy = lambda arr: RealishTensor(arr)

    try:
        full_tensor = RealishTensor(signal.reshape(1, -1))
        full = audio.apply_speed(full_tensor, speed=speed, sample_rate=sr).numpy()

        chunks = [
            RealishTensor(signal[start : start + 3200].reshape(1, -1))
            for start in range(0, signal.shape[0], 3200)
        ]
        streamed = np.concatenate(
            [
                chunk.numpy()
                for chunk in audio.stream_speed_chunks(chunks, speed=speed, sample_rate=sr)
            ],
            axis=1,
        )

        assert streamed.shape == full.shape
        assert np.allclose(streamed, full)
    finally:
        stub_torch.from_numpy = orig_from_numpy


def test_streaming_applies_speed_per_chunk(monkeypatch):
    """Streaming with speed != 1 should use generate_audio_stream, not generate_audio."""
    import app.routes as routes

    generate_audio_called = False
    stream_called = False

    class StreamCheckTTS(DummyTTS):
        def generate_audio(self, voice_state, text):
            nonlocal generate_audio_called
            generate_audio_called = True
            return audio.torch.Tensor((1, 48000))

        def generate_audio_stream(self, voice_state, text):
            nonlocal stream_called
            stream_called = True
            yield audio.torch.Tensor((1, 24000))

    # Stub apply_speed to avoid needing real numpy processing in route test
    monkeypatch.setattr(audio, 'apply_speed', lambda t, s, sr: t)
    # Stub tensor_to_pcm_bytes since DummyTensor doesn't support arithmetic
    monkeypatch.setattr(audio, 'tensor_to_pcm_bytes', lambda t: b'\x00\x00')
    monkeypatch.setattr(routes, 'tensor_to_pcm_bytes', lambda t: b'\x00\x00')

    monkeypatch.setattr(routes, 'get_tts_service', lambda: StreamCheckTTS())
    client = create_app({'TESTING': True, 'STREAM_DEFAULT': False}).test_client()

    response = client.post(
        '/v1/audio/speech',
        json={
            'input': 'hello',
            'voice': 'alba',
            'speed': 1.5,
            'response_format': 'pcm',
            'stream': True,
        },
    )

    assert response.status_code == 200
    assert stream_called, 'should use streaming generation'
    assert not generate_audio_called, 'should NOT fall back to full generation'
