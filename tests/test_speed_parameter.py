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

    def fake_apply_speed(audio_tensor, speed):
        calls['speed'] = speed
        return audio.torch.Tensor((1, 32000))

    monkeypatch.setattr(audio, 'apply_speed', fake_apply_speed)

    buffer = audio.convert_audio(audio.torch.Tensor((1, 48000)), 24000, 'wav', speed=1.5)

    assert calls == {'speed': 1.5}
    assert buffer.read() == b'wav:24000:32000'


def test_apply_speed_uses_phase_vocoder(monkeypatch):
    calls = {}
    spectrogram = audio.torch.Tensor((1, 513, 100))

    monkeypatch.setattr(audio.torch, 'stft', lambda *args, **kwargs: spectrogram)

    def fake_phase_vocoder(spectrogram_arg, rate, phase_advance):
        calls['rate'] = rate
        calls['phase_advance_shape'] = phase_advance.shape
        return audio.torch.Tensor((1, 513, 67))

    monkeypatch.setattr(audio.torchaudio.functional, 'phase_vocoder', fake_phase_vocoder)

    result = audio.apply_speed(audio.torch.Tensor((1, 48000)), 1.5)

    assert calls == {'rate': 1.5, 'phase_advance_shape': (513, 1)}
    assert result.shape == (1, 32000)
