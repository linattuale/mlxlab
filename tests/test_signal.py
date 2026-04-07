"""Tests for mlxlab.signal."""

import mlx.core as mx
import numpy as np
import pytest

import mlxlab as ml


# ---- fftfreq / rfftfreq -----------------------------------------------------

def test_fftfreq_even():
    f = ml.signal.fftfreq(8, d=1.0)
    expected = np.fft.fftfreq(8, d=1.0)
    for i in range(8):
        assert abs(f[i].item() - expected[i]) < 1e-6, f"fftfreq[{i}]"


def test_fftfreq_odd():
    f = ml.signal.fftfreq(7, d=0.5)
    expected = np.fft.fftfreq(7, d=0.5)
    for i in range(7):
        assert abs(f[i].item() - expected[i]) < 1e-6, f"fftfreq[{i}]"


def test_rfftfreq():
    f = ml.signal.rfftfreq(8, d=1.0)
    expected = np.fft.rfftfreq(8, d=1.0)
    assert f.shape[0] == expected.shape[0]
    for i in range(f.shape[0]):
        assert abs(f[i].item() - expected[i]) < 1e-6, f"rfftfreq[{i}]"


# ---- psd ---------------------------------------------------------------------

def test_psd_single_frequency():
    """PSD of a pure sine wave should peak at the right frequency."""
    fs = 1000.0
    freq = 50.0
    t = mx.arange(0, 1.0, 1.0 / fs)
    x = mx.sin(2.0 * 3.141592653589793 * freq * t)

    freqs, power = ml.signal.psd(x, fs=fs)
    # Peak should be near 50 Hz
    peak_idx = int(mx.argmax(power).item())
    peak_freq = freqs[peak_idx].item()
    assert abs(peak_freq - freq) < 2.0, f"PSD peak at {peak_freq} Hz, expected ~{freq} Hz"


def test_psd_shape():
    x = mx.random.normal((512,))
    freqs, power = ml.signal.psd(x, fs=100.0)
    assert freqs.shape[0] == 257  # 512//2 + 1
    assert power.shape[0] == 257


# ---- welch -------------------------------------------------------------------

def test_welch_reduces_variance():
    """Welch's method should produce smoother PSD than single-window."""
    mx.random.seed(42)
    x = mx.random.normal((4096,))
    _, p_single = ml.signal.psd(x, fs=1.0)
    _, p_welch = ml.signal.welch(x, fs=1.0, nperseg=256)

    # Welch should have lower variance (smoother)
    var_single = float(mx.var(p_single).item())
    var_welch = float(mx.var(p_welch).item())
    assert var_welch < var_single, f"Welch var {var_welch} >= single var {var_single}"


# ---- spectrogram -------------------------------------------------------------

def test_spectrogram_shape():
    x = mx.random.normal((2048,))
    times, freqs, Sxx = ml.signal.spectrogram(x, fs=100.0, nperseg=256, noverlap=128)
    n_segments = (2048 - 256) // 128 + 1  # 14
    n_freqs = 256 // 2 + 1  # 129
    assert Sxx.shape == (n_segments, n_freqs)
    assert times.shape[0] == n_segments
    assert freqs.shape[0] == n_freqs


def test_psd_odd_length():
    """Odd-length PSD should not halve the last bin (no Nyquist)."""
    # 7-sample signal: rfft returns 4 bins, none is Nyquist
    x = mx.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])
    freqs, power = ml.signal.psd(x, fs=1.0, window=False)
    assert freqs.shape[0] == 4  # n//2+1 = 4 for n=7
    # Last bin should NOT be halved (it's not Nyquist)
    # Verify by checking it's nonzero
    assert power[-1].item() > 0


def test_welch_invalid_overlap():
    with pytest.raises(ValueError, match="noverlap"):
        ml.signal.welch(mx.random.normal((1000,)), nperseg=256, noverlap=256)


def test_welch_short_signal():
    with pytest.raises(ValueError, match="Signal length"):
        ml.signal.welch(mx.random.normal((100,)), nperseg=256)


def test_spectrogram_invalid_nperseg():
    with pytest.raises(ValueError, match="nperseg"):
        ml.signal.spectrogram(mx.random.normal((1000,)), nperseg=0)


def test_spectrogram_chirp():
    """Spectrogram of a chirp should show increasing frequency over time."""
    fs = 1000.0
    t = mx.arange(0, 1.0, 1.0 / fs)
    # Linear chirp from 50 to 200 Hz
    freq = 50.0 + 150.0 * t
    phase = 2.0 * 3.141592653589793 * mx.cumsum(freq / fs)
    x = mx.sin(phase)

    times, freqs, Sxx = ml.signal.spectrogram(x, fs=fs, nperseg=128, noverlap=96)

    # Peak frequency should increase over time
    peak_freqs = []
    for i in range(Sxx.shape[0]):
        peak_idx = int(mx.argmax(Sxx[i]).item())
        peak_freqs.append(freqs[peak_idx].item())

    # First quarter should have lower peak than last quarter
    n = len(peak_freqs)
    early = np.mean(peak_freqs[:n // 4])
    late = np.mean(peak_freqs[-n // 4:])
    assert late > early, f"Chirp: early peak {early:.0f} Hz >= late peak {late:.0f} Hz"
