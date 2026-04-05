"""
Audio conversion and streaming utilities.
"""

import io
import struct

import numpy as np
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


def _wsola_stretch(audio: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    """
    WSOLA (Waveform Similarity Overlap-Add) time-stretching for a mono signal.

    Produces much cleaner speech output than a phase vocoder because it works
    directly in the time domain and avoids the spectral smearing / "phasiness"
    artefacts inherent to STFT-based approaches.

    Args:
        audio: 1-D float array of mono samples
        speed: Playback speed multiplier (>1 = faster / shorter)
        sample_rate: Sample rate in Hz (used to size the analysis window)

    Returns:
        Time-stretched mono audio array
    """
    n = len(audio)

    # Window / hop sizes in samples (50 ms frame, 12.5 ms synthesis hop)
    frame_len = max(4, int(0.050 * sample_rate))
    syn_hop = max(1, int(0.0125 * sample_rate))
    ana_hop = max(1, int(syn_hop * speed))
    tolerance = max(0, int(0.005 * sample_rate))  # 5 ms cross-correlation search

    if n < frame_len:
        # Too short to process – return as-is
        return audio

    window = np.hanning(frame_len).astype(np.float64)

    n_frames = max(1, (n - frame_len) // ana_hop + 1)
    out_len = (n_frames - 1) * syn_hop + frame_len
    output = np.zeros(out_len, dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)

    delta = 0  # cumulative position offset from cross-correlation

    for i in range(n_frames):
        ana_start = max(0, min(n - frame_len, i * ana_hop + delta))
        syn_start = i * syn_hop

        # Cross-correlation search for best overlap (skip first frame)
        if i > 0 and tolerance > 0:
            lo = max(0, ana_start - tolerance)
            hi = min(n - frame_len, ana_start + tolerance)
            if lo < hi:
                ref = output[syn_start : syn_start + frame_len].copy()
                ref_norm = norm[syn_start : syn_start + frame_len].copy()
                ref_norm[ref_norm < 1e-8] = 1.0
                ref /= ref_norm
                ref_w = ref * window

                search_region = audio[lo : hi + frame_len]
                candidates = np.lib.stride_tricks.sliding_window_view(search_region, frame_len)[
                    : hi - lo + 1
                ]
                scores = candidates @ ref_w
                best_pos = lo + int(np.argmax(scores))
                delta += best_pos - ana_start
                ana_start = best_pos

        frame = audio[ana_start : ana_start + frame_len] * window
        output[syn_start : syn_start + frame_len] += frame
        norm[syn_start : syn_start + frame_len] += window

    norm[norm < 1e-8] = 1.0
    output /= norm
    return output.astype(np.float32)


def apply_speed(
    audio_tensor: torch.Tensor, speed: float = 1.0, sample_rate: int = 24000
) -> torch.Tensor:
    """
    Apply pitch-preserving time stretch to synthesized audio.

    Uses WSOLA (Waveform Similarity Overlap-Add) which works in the time
    domain, avoiding the spectral smearing ("fuzzy" / metallic sound) that
    a phase-vocoder approach introduces on speech.

    Args:
        audio_tensor: The audio waveform (1D or 2D)
        speed: Playback speed multiplier
        sample_rate: Sample rate of the audio in Hz

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

    device = audio_tensor.device
    dtype = audio_tensor.dtype
    cpu_tensor = audio_tensor.cpu() if audio_tensor.is_cuda else audio_tensor

    channels = []
    for ch in range(cpu_tensor.shape[0]):
        mono = cpu_tensor[ch].numpy().astype(np.float64)
        stretched = _wsola_stretch(mono, speed, sample_rate)
        channels.append(stretched)

    result = torch.from_numpy(np.stack(channels)).to(dtype=dtype, device=device)
    return result.squeeze(0) if was_1d else result


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

    audio_tensor = apply_speed(audio_tensor, speed, sample_rate)

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
