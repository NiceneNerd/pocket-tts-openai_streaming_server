"""
Audio conversion and streaming utilities.
"""

import io
import math
import struct

import torch
import torchaudio

from app.logging_config import get_logger

logger = get_logger('audio')

# Valid audio formats
VALID_FORMATS = {'mp3', 'wav', 'opus', 'aac', 'flac', 'pcm'}
MIN_SPEED = 0.25
MAX_SPEED = 4.0


def validate_format(fmt: str) -> str:
    """
    Normalize and validate the requested audio format.

    Args:
        fmt: Requested format string

    Returns:
        Validated format string
    """
    fmt = fmt.lower()

    # OpenAI sometimes sends 'mpeg' for mp3
    if fmt == 'mpeg':
        return 'mp3'

    if fmt not in VALID_FORMATS:
        logger.warning(f"Unknown format '{fmt}', falling back to wav")
        return 'wav'

    return fmt


def validate_speed(speed: float | int | str | None) -> float:
    """
    Normalize and validate the requested playback speed.

    Args:
        speed: Requested speed value

    Returns:
        Validated speed as a float

    Raises:
        ValueError: If the speed is not numeric or out of range
    """
    if speed is None:
        return 1.0

    if isinstance(speed, bool):
        raise ValueError("'speed' must be a number")

    try:
        speed = float(speed)
    except (TypeError, ValueError) as exc:
        raise ValueError("'speed' must be a number") from exc

    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"'speed' must be between {MIN_SPEED} and {MAX_SPEED}")

    return speed


def apply_speed(audio_tensor: torch.Tensor, speed: float = 1.0) -> torch.Tensor:
    """
    Apply pitch-preserving time stretch to synthesized audio.

    Uses a phase vocoder so changing playback speed does not also raise the
    perceived pitch into a "chipmunk" voice.

    Args:
        audio_tensor: The audio waveform (1D or 2D)
        speed: Playback speed multiplier

    Returns:
        Audio tensor with adjusted duration and preserved pitch
    """
    speed = validate_speed(speed)
    if speed == 1.0:
        return audio_tensor

    was_1d = audio_tensor.dim() == 1
    if was_1d:
        audio_tensor = audio_tensor.unsqueeze(0)

    num_samples = audio_tensor.shape[-1]
    if num_samples < 2:
        return audio_tensor.squeeze(0) if was_1d else audio_tensor

    n_fft = min(1024, 2 ** int(math.floor(math.log2(num_samples))))
    hop_length = max(1, n_fft // 4)
    window = torch.hann_window(n_fft, device=audio_tensor.device, dtype=audio_tensor.dtype)

    spectrogram = torch.stft(
        audio_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    phase_advance = torch.linspace(
        0,
        math.pi * hop_length,
        spectrogram.size(-2),
        device=spectrogram.device,
        dtype=audio_tensor.dtype,
    ).unsqueeze(-1)
    stretched = torchaudio.functional.phase_vocoder(
        spectrogram,
        rate=speed,
        phase_advance=phase_advance,
    )

    audio_tensor = torch.istft(
        stretched,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        length=max(1, int(round(num_samples / speed))),
    )

    return audio_tensor.squeeze(0) if was_1d else audio_tensor


def convert_audio(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    target_format: str = 'wav',
    speed: float = 1.0,
) -> io.BytesIO:
    """
    Convert a raw audio tensor to a byte buffer in the specified format.

    Args:
        audio_tensor: The audio waveform (1D or 2D)
        sample_rate: The sample rate of the audio
        target_format: The target audio format
        speed: Playback speed multiplier

    Returns:
        Buffer containing the encoded audio data
    """
    buffer = io.BytesIO()

    audio_tensor = apply_speed(audio_tensor, speed)

    # Ensure tensor is CPU
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()

    # Ensure 2D (channels, time)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    try:
        torchaudio.save(buffer, audio_tensor, sample_rate, format=target_format)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f'Error converting audio to {target_format}: {e}')
        raise


def write_wav_header(
    sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16, num_frames: int = 0
) -> bytes:
    """
    Generate a WAV header for streaming.

    If num_frames is 0, set to max value (streaming/unknown length).

    Args:
        sample_rate: Audio sample rate
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample
        num_frames: Number of frames (0 for unknown/streaming)

    Returns:
        WAV header bytes
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    # Data size: if unknown, max uint32
    data_size = num_frames * block_align
    if num_frames == 0:
        data_size = 0xFFFFFFFF - 36

    chunk_size = 36 + data_size

    header = io.BytesIO()
    header.write(b'RIFF')
    header.write(struct.pack('<I', chunk_size))
    header.write(b'WAVE')
    header.write(b'fmt ')
    header.write(struct.pack('<I', 16))  # Subchunk1Size (16 for PCM)
    header.write(struct.pack('<H', 1))  # AudioFormat (1 for PCM)
    header.write(struct.pack('<H', num_channels))
    header.write(struct.pack('<I', sample_rate))
    header.write(struct.pack('<I', byte_rate))
    header.write(struct.pack('<H', block_align))
    header.write(struct.pack('<H', bits_per_sample))
    header.write(b'data')
    header.write(struct.pack('<I', data_size))

    return header.getvalue()


def tensor_to_pcm_bytes(chunk_tensor: torch.Tensor) -> bytes:
    """
    Convert audio tensor chunk to 16-bit PCM bytes.

    Args:
        chunk_tensor: Audio tensor chunk

    Returns:
        PCM audio bytes
    """
    if chunk_tensor.is_cuda:
        chunk_tensor = chunk_tensor.cpu()

    if chunk_tensor.dim() == 1:
        chunk_tensor = chunk_tensor.unsqueeze(0)

    # Convert to 16-bit PCM
    pcm = (chunk_tensor * 32767).clamp(-32768, 32767).to(torch.int16)
    return pcm.numpy().tobytes()


def get_mime_type(fmt: str) -> str:
    """
    Get the MIME type for an audio format.

    Args:
        fmt: Audio format string

    Returns:
        MIME type string
    """
    mime_types = {
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg',
        'pcm': 'audio/L16',
        'opus': 'audio/opus',
        'aac': 'audio/aac',
        'flac': 'audio/flac',
    }
    return mime_types.get(fmt, f'audio/{fmt}')
