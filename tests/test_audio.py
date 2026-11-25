import numpy as np
import pytest
from backend.audio import pcm16_to_float32, calculate_rms_energy, BoundedAudioBuffer

def test_pcm16_to_float32_conversion():
    # 0 in int16 -> 0.0 in float32
    # 32767 in int16 -> ~1.0 in float32
    # -32768 in int16 -> -1.0 in float32
    pcm_bytes = (0).to_bytes(2, byteorder="little", signed=True) + \
                (32767).to_bytes(2, byteorder="little", signed=True) + \
                (-32768).to_bytes(2, byteorder="little", signed=True)
    
    float_arr = pcm16_to_float32(pcm_bytes)
    assert len(float_arr) == 3
    assert np.isclose(float_arr[0], 0.0, atol=1e-4)
    assert np.isclose(float_arr[1], 1.0, atol=1e-4)
    assert np.isclose(float_arr[2], -1.0, atol=1e-4)

def test_calculate_rms_energy_silence_vs_sound():
    silence = b"\x00" * 3200
    sound = (10000).to_bytes(2, byteorder="little", signed=True) * 1600
    
    rms_silence = calculate_rms_energy(silence)
    rms_sound = calculate_rms_energy(sound)
    
    assert rms_silence == 0.0
    assert rms_sound > 5000.0

def test_bounded_audio_buffer_bounds():
    # 1 second of 16kHz 16-bit audio = 32,000 bytes
    buffer = BoundedAudioBuffer(sample_rate=16000, max_seconds=2.0)
    assert buffer.duration_seconds() == 0.0
    
    # Append 3 seconds of audio
    three_seconds_bytes = b"\x01\x00" * (16000 * 3)
    buffer.append(three_seconds_bytes)
    
    # Must be bounded to 2.0 seconds
    assert buffer.duration_seconds() == pytest.approx(2.0, abs=0.01)
    
    # Test audio retrieval
    arr = buffer.get_audio_float32()
    assert len(arr) == 16000 * 2

def test_bounded_audio_buffer_trim():
    buffer = BoundedAudioBuffer(sample_rate=16000, max_seconds=5.0)
    two_sec_bytes = b"\x01\x00" * (16000 * 2)
    buffer.append(two_sec_bytes)
    
    # Trim 1.0 second with 0.2s overlap -> removes 0.8 seconds
    buffer.trim_front(seconds_to_trim=1.0, keep_overlap_sec=0.2)
    assert buffer.duration_seconds() == pytest.approx(1.2, abs=0.01)
